package haiqing.word2vec;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jblas.NDArray;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class FirstIterationFunction
        implements FlatMapFunction< Iterator<List<VocabWord>>, Entry<Pair<Integer,Integer>, INDArray> > {

    private int ithIteration = 1;
    private int vectorLength;
    private boolean useAdaGrad;
    private int negative;
    private int window;
    private double alpha;
    private double minAlpha;
    private long totalWordCount;
    private long seed;
    private int maxExp;
    private double[] expTable;
    private Broadcast<Map<Pair<Integer,Integer>, INDArray>> syn0;
    private VocabCache vocabCache;
    private int K;

    private Map<Pair<Integer,Integer>, INDArray> indexSyn0VecMap;
    private AtomicLong nextRandom = new AtomicLong(5);
    private int vecNum;


    public FirstIterationFunction(Map<String, Object> word2vecVarMap,
                                  double[] expTable, Broadcast<Map<Pair<Integer,Integer>, INDArray>> syn0, VocabCache vocabCache) {

        this.expTable = expTable;
        this.vectorLength = Integer.parseInt(word2vecVarMap.get("vectorLength").toString());
        this.useAdaGrad = Boolean.getBoolean(word2vecVarMap.get("useAdaGrad").toString());
        this.negative = Integer.parseInt(word2vecVarMap.get("negative").toString());
        this.window = Integer.parseInt(word2vecVarMap.get("window").toString());
        this.alpha = Double.parseDouble(word2vecVarMap.get("alpha").toString());
        this.minAlpha = Double.parseDouble(word2vecVarMap.get("minAlpha").toString());
        this.totalWordCount = Long.parseLong(word2vecVarMap.get("totalWordCount").toString());
        this.seed = Long.parseLong(word2vecVarMap.get("seed").toString());
        this.maxExp = Integer.parseInt(word2vecVarMap.get("maxExp").toString());
        this.vecNum = Integer.parseInt(word2vecVarMap.get("vecNum").toString());
        this.K = Integer.parseInt(word2vecVarMap.get("K").toString());
        this.syn0 = syn0;
        this.vocabCache = vocabCache;
    }

    public Iterable<Entry<Pair<Integer,Integer>, INDArray>> call(Iterator<List<VocabWord>> iter) {
        indexSyn0VecMap = syn0.value();
        Long last = 0L;
        Long now = 0L;

        while (iter.hasNext()) {
            List<VocabWord> vocabWordsList = iter.next();
            double currentSentenceAlpha = Math.max(minAlpha,
                    alpha - (alpha - minAlpha) * (now / (double) totalWordCount));
            if (now-last > 10000) {
                System.out.println("sentenceCumSumCount: " + now + "   currentSentenceAlpha: " + currentSentenceAlpha);
                last = now;
            }
            trainSentence(vocabWordsList, currentSentenceAlpha);
            now += vocabWordsList.size();
        }

        //indexSyn0VecMap.put(new Pair(0,0), Nd4j.rand(1,vectorLength));
        return indexSyn0VecMap.entrySet();
    }

    public void trainSentence(List<VocabWord> vocabWordsList, double currentSentenceAlpha) {

        if (vocabWordsList != null && !vocabWordsList.isEmpty()) {
            for (int ithWordInSentence = 0; ithWordInSentence < vocabWordsList.size(); ithWordInSentence++) {
                // Random value ranging from 0 to window size
                nextRandom.set(nextRandom.get() * 25214903917L + 11);
                int b = (int) (long) this.nextRandom.get() % window;
                VocabWord currentWord = vocabWordsList.get(ithWordInSentence);
                if (currentWord != null) {
                    skipGram(ithWordInSentence, vocabWordsList, b, currentSentenceAlpha);
                }
            }
        }
    }

    public void skipGram(int ithWordInSentence, List<VocabWord> vocabWordsList, int b, double currentSentenceAlpha) {

        VocabWord currentWord = vocabWordsList.get(ithWordInSentence);
        if (currentWord != null && !vocabWordsList.isEmpty()) {
            int end = window * 2 + 1 - b;
            for (int a = b; a < end; a++) {
                if (a != window) {
                    int c = ithWordInSentence - window + a;
                    if (c >= 0 && c < vocabWordsList.size()) {
                        VocabWord lastWord = vocabWordsList.get(c);
                        if (currentWord != null && lastWord != null)
                            iterateSample(vocabCache.wordFor(currentWord.getWord()), vocabCache.wordFor(lastWord.getWord()), currentSentenceAlpha);
                    }
                }
            }
        }
    }

    public void iterateSample(VocabWord w1, VocabWord w2, double currentSentenceAlpha) {

        if (w2 == null || w2.getIndex() < 0 || w1.getIndex() == w2.getIndex())
            return;

        // error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);

        Pair<Integer,Integer> w2Index = null;
        double maxScore = -1;
        for (int k = 0; k < K; k++) {
            Pair<Integer,Integer> w2IndexK = new Pair(w2.getIndex(),k);
            INDArray syn0Vec = indexSyn0VecMap.get(w2IndexK);
            double score = 1;
            for (int i = 0; i < w1.getCodeLength(); i++) {
                int point = w1.getPoints().get(i) + vecNum;
                Pair<Integer,Integer> pointIndex = new Pair(point,0);
                INDArray syn1Vec = indexSyn0VecMap.get(pointIndex);
                double dot = Nd4j.getBlasWrapper().level1().dot(vectorLength, 1.0, syn0Vec, syn1Vec);
                if (dot < -maxExp || dot >= maxExp)
                    continue;
                int idx = (int) ((dot + maxExp) * ((double) expTable.length / maxExp / 2.0));
                //score
                double f = expTable[idx];
                score *= f;
            }
            if (score > maxScore) {
                maxScore = score;
                w2Index = w2IndexK;
            }
        }

        INDArray syn0Vec = indexSyn0VecMap.get(w2Index);
        //
        for (int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i)+vecNum;

            // Point to
            Pair<Integer,Integer> pointIndex = new Pair(point,0);
            INDArray syn1Vec = indexSyn0VecMap.get(pointIndex);

            // Dot product of Syn0 and Syn1 vecs
            double dot = Nd4j.getBlasWrapper().level1().dot(vectorLength, 1.0, syn0Vec, syn1Vec);

            if (dot < -maxExp || dot >= maxExp)
                continue;

            int idx = (int) ((dot + maxExp) * ((double) expTable.length / maxExp / 2.0));

            //score
            double f = expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ? w1.getGradient(i, currentSentenceAlpha) : currentSentenceAlpha);

            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, syn1Vec, neu1e);
            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, syn0Vec, syn1Vec);

            indexSyn0VecMap.put(pointIndex, syn1Vec);
        }

        // Updated the Syn0 vector based on gradient. Syn0 is not random anymore.
        Nd4j.getBlasWrapper().level1().axpy(vectorLength, 1.0f, neu1e, syn0Vec);

        indexSyn0VecMap.put(w2Index, syn0Vec);

    }
}
