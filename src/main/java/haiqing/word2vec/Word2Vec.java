package haiqing.word2vec;

import com.google.common.collect.Lists;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.*;
import java.util.*;

/**
 * Spark version of word2vec
 *
 * @author Adam Gibson
 */
public class Word2Vec implements Serializable  {

    private INDArray trainedSyn1;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
    private int MAX_EXP = 6;
    private double[] expTable;

    // Input by user only via setters
    private int vectorLength = 100;
    private boolean useAdaGrad = false;
    private int negative = 0;
    private int numWords = 1;
    private int window = 5;
    private double alpha= 0.025;
    private double minAlpha = 0.0001;
    private int numPartitions = 1;
    private int iterations = 1;
    private int nGrams = 1;
    private String tokenizer = "org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory";
    private String tokenPreprocessor = "org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor";
    private boolean removeStop = false;
    private long seed = 42L;
    private int K = 1;
    private VocabCache vocab;
    private INDArray syn0;
    private String path = "vectors.txt";

    // Constructor to take InMemoryLookupCache table from an already trained model
    public Word2Vec(INDArray trainedSyn1) {
        this.trainedSyn1 = trainedSyn1;
        this.expTable = initExpTable();
    }

    public Word2Vec() {
        this.expTable = initExpTable();
    }

    public double[] initExpTable() {
        double[] expTable = new double[1000];
        for (int i = 0; i < expTable.length; i++) {
            double tmp = FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i] = tmp / (tmp + 1.0);
        }
        return expTable;
    }

    public Map<String, Object> getTokenizerVarMap() {
        return new HashMap<String, Object>() {{
            put("numWords", numWords);
            put("nGrams", nGrams);
            put("tokenizer", tokenizer);
            put("tokenPreprocessor", tokenPreprocessor);
            put("removeStop", removeStop);
        }};
    }

    public Map<String, Object> getWord2vecVarMap() {
        return new HashMap<String, Object>() {{
            put("vectorLength", vectorLength);
            put("useAdaGrad", useAdaGrad);
            put("negative", negative);
            put("window", window);
            put("alpha", alpha);
            put("minAlpha", minAlpha);
            put("iterations", iterations);
            put("seed", seed);
            put("maxExp", MAX_EXP);
            put("K", K);
        }};
    }

    public class MapPairFunction implements PairFunction<Map.Entry<Integer, INDArray>, Integer, INDArray> {
        public Tuple2<Integer, INDArray> call(Map.Entry<Integer, INDArray> pair) {
            return new Tuple2(pair.getKey(), pair.getValue());
        }
    }

    public class Sum implements Function2<INDArray, INDArray, INDArray> {
        public INDArray call(INDArray a, INDArray b) {
            return a.add(b);
        }
    }

    // Training word2vec based on corpus
    public void train(JavaRDD<String> corpusRDD) throws Exception {
        log.info("Start training ...");


        // SparkContext
        final JavaSparkContext sc = new JavaSparkContext(corpusRDD.context());

        // Pre-defined variables
        Map<String, Object> tokenizerVarMap = getTokenizerVarMap();
        Map<String, Object> word2vecVarMap = getWord2vecVarMap();

        // Variables to fill in in train
        //final JavaRDD<AtomicLong> sentenceWordsCountRDD;
        final JavaRDD<List<VocabWord>> vocabWordListRDD;
        //final JavaPairRDD<List<VocabWord>, Long> vocabWordListSentenceCumSumRDD;
        final VocabCache vocabCache;
        final JavaRDD<Long> sentenceCumSumCountRDD;

        // Start Training //
        //////////////////////////////////////
        log.info("Tokenization and building VocabCache ...");
        // Processing every sentence and make a VocabCache which gets fed into a LookupCache
        Broadcast<Map<String, Object>> broadcastTokenizerVarMap = sc.broadcast(tokenizerVarMap);
        TextPipeline pipeline = new TextPipeline(corpusRDD.repartition(numPartitions), broadcastTokenizerVarMap);
        pipeline.buildVocabCache();
        pipeline.buildVocabWordListRDD();

        // Get total word count and put into word2vec variable map
        word2vecVarMap.put("totalWordCount", pipeline.getTotalWordCount()/numPartitions);

        // 2 RDDs: (vocab words list) and (sentence Count).Already cached
        //sentenceWordsCountRDD = pipeline.getSentenceCountRDD();
        vocabWordListRDD = pipeline.getVocabWordListRDD();
        System.out.println("-------");
        Iterator a = vocabWordListRDD.collect().iterator();
        for (int i = 0; i < 10; i++)
            System.out.println(a.next().toString());

        // Get vocabCache and broad-casted vocabCache
        Broadcast<VocabCache> vocabCacheBroadcast = pipeline.getBroadCastVocabCache();
        vocabCache = vocabCacheBroadcast.getValue();

        //////////////////////////////////////
        log.info("Building Huffman Tree ...");
        // Building Huffman Tree would update the code and point in each of the vocabWord in vocabCache


        System.out.println("-------");
        a = vocabCache.vocabWords().iterator();
        for (int i = 0; i < 10; i++)
            System.out.println(a.next().toString());
        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();
        System.out.println("-------");
        a = vocabCache.vocabWords().iterator();
        for (int i = 0; i < 10; i++)
            System.out.println(a.next().toString());


        System.out.println("-------");
        a = vocabWordListRDD.collect().iterator();
        for (int i = 0; i < 10; i++)
            System.out.println(a.next().toString());

        /////////////////////////////////////
        log.info("Training word2vec sentences ...");

        word2vecVarMap.put("vecNum", vocabCache.numWords());

        //Map<Tuple2<Integer,Integer>, INDArray> s0 = new HashMap();
        Map<Pair<Integer,Integer>, INDArray> s0 = new HashMap();
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < vocabCache.numWords(); i++) {
                s0.put(new Pair(i, k), getRandomSyn0Vec(vectorLength));
            }
        }
        for (int i = vocabCache.numWords(); i < vocabCache.numWords()*2-1; i++) {
            s0.put(new Pair(i,0), Nd4j.zeros(1, vectorLength));
        }

        System.out.println(s0.get(new Pair<Integer, Integer>(0,0)).toString());

        for (int i = 0; i < iterations; i++) {
            System.out.println("iteration: "+i);

            word2vecVarMap.put("alpha", alpha-(alpha-minAlpha)/iterations*i);
            word2vecVarMap.put("minAlpha", alpha-(alpha-minAlpha)/iterations*(i+1));

            FlatMapFunction firstIterationFunction = new FirstIterationFunction(word2vecVarMap, expTable, sc.broadcast(s0), vocabCache);

            //@SuppressWarnings("unchecked")
            JavaPairRDD<Pair<Integer,Integer>, INDArray> indexSyn0UpdateEntryRDD =
                    vocabWordListRDD.mapPartitions(firstIterationFunction).mapToPair(new MapPairFunction()).cache();
            Map<Pair<Integer,Integer>, Object> count = indexSyn0UpdateEntryRDD.countByKey();
            indexSyn0UpdateEntryRDD = indexSyn0UpdateEntryRDD.reduceByKey(new Sum());

            // Get all the syn0 updates into a list in driver
            List<Tuple2<Pair<Integer,Integer>, INDArray>> syn0UpdateEntries = indexSyn0UpdateEntryRDD.collect();



            // Updating syn0
            s0 = new HashMap();
            for (Tuple2<Pair<Integer,Integer>, INDArray> syn0UpdateEntry : syn0UpdateEntries) {
                int cc = Integer.parseInt(count.get(syn0UpdateEntry._1()).toString());
                //int cc = 1;
                if (cc > 0) {
                    INDArray tmp = Nd4j.zeros(1, vectorLength).addi(syn0UpdateEntry._2()).divi(cc);
                    s0.put(syn0UpdateEntry._1(), tmp);
                    if (syn0UpdateEntry._1().getFirst() == 0 && syn0UpdateEntry._1().getSecond() == 0)
                        System.out.println(tmp.toString());
                }
            }
            //System.out.println(s0.get(new Pair<Integer, Integer>(0,0)).toString());
        }

        syn0 = Nd4j.zeros(vocabCache.numWords()*K, vectorLength);
        for (Map.Entry<Pair<Integer,Integer>, INDArray> ss: s0.entrySet()) {
            if (ss.getKey().getFirst() < vocabCache.numWords()) {
                syn0.getRow(ss.getKey().getSecond()*vocabCache.numWords()+ss.getKey().getFirst()).addi(ss.getValue());
            }
        }

        vocab = vocabCache;
        syn0.diviRowVector(syn0.norm2(1));

        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path), false));
        for (int i = 0; i < syn0.rows(); i++) {
            String word = vocab.wordAtIndex(i%vocab.numWords());
            if (word == null) {
                continue;
            }
            word = word+"("+i/vocab.numWords()+")";
            StringBuilder sb = new StringBuilder();
            sb.append(word.replaceAll(" ", "_"));
            sb.append(" ");
            INDArray wordVector = syn0.getRow(i);
            for (int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if (j < wordVector.length() - 1) {
                    sb.append(" ");
                }
            }
            sb.append("\n");
            write.write(sb.toString());
        }
        write.flush();
        write.close();
    }

    public static INDArray readVocab(InMemoryLookupCache vocab, String path, int K) throws IOException{
        BufferedReader br = new BufferedReader(new FileReader(path));
        Map<Pair<Integer,Integer>, INDArray> s0 = new HashMap();
        int num = 0;
        int vectorLength = 0;
        try {
            int n = 0;
            int l = 0;
            while (true) {
                String line = br.readLine();
                if (line == null)
                    break;
                String[] ss = line.split(" ");
                String word = ss[0].substring(0, ss[0].length() - 3);
                int k = Integer.parseInt(ss[0].substring(ss[0].length() - 2, ss[0].length() - 1));
                double[] vector = new double[ss.length-1];
                vectorLength = ss.length-1;
                for (int i = 1; i < ss.length; i++)
                    vector[i-1] = Double.parseDouble(ss[i]);
                s0.put(new Pair(n, k), Nd4j.create(vector));
                if (k == 0)
                    addTokenToVocabCache(vocab,word);
                n++;
                if (k > l) {
                    num = n;
                    n = 0;
                    l = k;
                }
            }
        } finally {
            br.close();
        }
        INDArray syn0 = Nd4j.zeros(num*K, vectorLength);
        for (Map.Entry<Pair<Integer,Integer>, INDArray> ss: s0.entrySet()) {
            if (ss.getKey().getFirst() < num) {
                syn0.getRow(ss.getKey().getSecond()*num+ss.getKey().getFirst()).addi(ss.getValue());
            }
        }
        return syn0;
    }

    private static void addTokenToVocabCache(InMemoryLookupCache vocab, String stringToken) {
        // Making string token into actual token if not already an actual token (vocabWord)
        VocabWord actualToken;
        if (vocab.hasToken(stringToken)) {
            actualToken = vocab.tokenFor(stringToken);
        } else {
            actualToken = new VocabWord(1, stringToken);
        }

        // Set the index of the actual token (vocabWord)
        // Put vocabWord into vocabs in InMemoryVocabCache
        boolean vocabContainsWord = vocab.containsWord(stringToken);
        if (!vocabContainsWord) {
            vocab.addToken(actualToken);
            int idx = vocab.numWords();
            actualToken.setIndex(idx);
            vocab.putVocabWord(stringToken);
        }
    }


    public Collection<String> wordsNearest(String word, int k, int n) {
        INDArray vector = Transforms.unitVec(getWordVectorMatrix(word, k));
        INDArray similarity = vector.mmul(syn0.transpose());
        List<Double> highToLowSimList = getTopN(similarity, n);
        List<String> ret = new ArrayList();

        for (int i = 1; i < highToLowSimList.size(); i++) {
            word = vocab.wordAtIndex(highToLowSimList.get(i).intValue()%vocab.numWords())+"("+highToLowSimList.get(i).intValue()/vocab.numWords()+")";
            if (word != null && !word.equals("UNK") && !word.equals("STOP")) {
                ret.add(word);
                if (ret.size() >= n) {
                    break;
                }
            }
        }

        return ret;
    }

    public static Collection<String> wordsNearest(INDArray syn0, InMemoryLookupCache vocab, String word, int k, int n, int K) {


        INDArray vector = Transforms.unitVec(getWordVectorMatrix(syn0, vocab, word, k, K));
        INDArray similarity = vector.mmul(syn0.transpose());
        List<Double> highToLowSimList = getTopN(similarity, n);
        List<String> ret = new ArrayList();

        for (int i = 1; i < highToLowSimList.size(); i++) {
            word = vocab.wordAtIndex(highToLowSimList.get(i).intValue()%vocab.numWords())+"("+highToLowSimList.get(i).intValue()/vocab.numWords()+")";
            if (word != null && !word.equals("UNK") && !word.equals("STOP")) {
                ret.add(word);
                if (ret.size() >= n) {
                    break;
                }
            }
        }

        return ret;
    }

    public static INDArray getWordVectorMatrix(INDArray syn0, InMemoryLookupCache vocab, String word, int k, int K) {
        if(word == null || k > K)
            return null;
        int idx = vocab.indexOf(word);
        if(idx < 0)
            idx = vocab.indexOf(org.deeplearning4j.models.word2vec.Word2Vec.UNK);
        return syn0.getRow(vocab.numWords()*k+idx);
    }

    private static class ArrayComparator implements Comparator<Double[]> {
        public int compare(Double[] o1, Double[] o2) {
            return Double.compare(o1[0], o2[0]);
        }

    }

    private static List<Double> getTopN(INDArray vec, int N) {
        ArrayComparator comparator = new ArrayComparator();
        PriorityQueue<Double[]> queue = new PriorityQueue(vec.rows(), comparator);

        for (int j = 0; j < vec.length(); j++) {
            final Double[] pair = new Double[]{vec.getDouble(j), (double) j};
            if (queue.size() < N) {
                queue.add(pair);
            } else {
                Double[] head = queue.peek();
                if (comparator.compare(pair, head) > 0) {
                    queue.poll();
                    queue.add(pair);
                }
            }
        }
        List<Double> lowToHighSimLst = new ArrayList();

        while (!queue.isEmpty()) {
            double ind = queue.poll()[1];
            lowToHighSimLst.add(ind);
        }
        return Lists.reverse(lowToHighSimLst);
    }

    public double similarity(String word, int k1, String word2, int k2) {
        if (k1 > K || k2 > K)
            return -1;

        if(word.equals(word2) && k1 == k2)
            return 1.0;

        INDArray vector = Transforms.unitVec(getWordVectorMatrix(word, k1));
        INDArray vector2 = Transforms.unitVec(getWordVectorMatrix(word2, k2));
        if(vector == null || vector2 == null)
            return -1;
        return  Nd4j.getBlasWrapper().dot(vector, vector2);
    }

    public INDArray getWordVectorMatrix(String word, int k) {
        if(word == null || k > K)
            return null;
        int idx = vocab.indexOf(word);
        if(idx < 0)
            idx = vocab.indexOf(org.deeplearning4j.models.word2vec.Word2Vec.UNK);
        return syn0.getRow(vocab.numWords()*k+idx);
    }

    public INDArray getRandomSyn0Vec(int vectorLength) {
        return Nd4j.rand(seed, new int[]{1 ,vectorLength}).subi(0.5).divi(vectorLength);
    }

    public int getVectorLength() {
        return vectorLength;
    }

    public Word2Vec setVectorLength(int vectorLength) {
        this.vectorLength = vectorLength;
        return this;
    }

    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public Word2Vec setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
        return this;
    }

    public int getNegative() {
        return negative;
    }

    public Word2Vec setNegative(int negative) {
        this.negative = negative;
        return this;
    }

    public int getNumWords() {
        return numWords;
    }

    public Word2Vec setNumWords(int numWords) {
        this.numWords = numWords;
        return this;
    }

    public int getWindow() {
        return window;
    }

    public Word2Vec setWindow(int window) {
        this.window = window;
        return this;
    }

    public double getAlpha() {
        return alpha;
    }

    public Word2Vec setAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }

    public Word2Vec setNumPartitions(int numPartitions) {
        this.numPartitions = numPartitions;
        return this;
    }

    public double getMinAlpha() {
        return minAlpha;
    }

    public Word2Vec setMinAlpha(double minAlpha) {
        this.minAlpha = minAlpha;
        return this;
    }

    public int getIterations() {
        return iterations;
    }

    public Word2Vec setIterations(int iterations) {
        this.iterations = iterations;
        return this;
    }

    public int getnGrams() {
        return nGrams;
    }

    public Word2Vec setnGrams(int nGrams) {
        this.nGrams = nGrams;
        return this;
    }

    public String getTokenizer() {
        return tokenizer;
    }

    public Word2Vec setTokenizer(String tokenizer) {
        this.tokenizer = tokenizer;
        return this;
    }

    public String getTokenPreprocessor() {
        return tokenPreprocessor;
    }

    public Word2Vec setTokenPreprocessor(String tokenPreprocessor) {
        this.tokenPreprocessor = tokenPreprocessor;
        return this;
    }

    public boolean isRemoveStop() {
        return removeStop;
    }

    public Word2Vec setRemoveStop(boolean removeStop) {
        this.removeStop = removeStop;
        return this;
    }

    public long getSeed() {
        return seed;
    }

    public Word2Vec setSeed(long seed) {
        this.seed = seed;
        return this;
    }

    public double[] getExpTable() {
        return expTable;
    }
}