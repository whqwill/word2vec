package haiqing.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * Created by hwang on 05.11.15.
 */
public class testW2V {
    private static Logger log = LoggerFactory.getLogger(testW2V.class);

    public static void main(String[] args) throws Exception {
        long begintime = System.currentTimeMillis();

        SparkConf sparkConf = new SparkConf().setAppName("sparktest");
        if (!sparkConf.contains("spark.master")) {
            sparkConf.setMaster("local[*]");
        }
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        System.out.println(sc.defaultParallelism() + "   " + sc.master());

        JavaRDD<String> corpus = sc.textFile("raw_sentences.txt");

        Word2Vec word2Vec = new Word2Vec()
                .setnGrams(1)
                .setTokenizer("org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory")
                        //.setTokenPreprocessor("Preprocessor")
                .setTokenPreprocessor("org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor")
                .setRemoveStop(false)
                .setSeed(42L)
                .setNegative(0)
                .setUseAdaGrad(false)
                .setVectorLength(100)
                .setWindow(5)
                .setAlpha(0.025).setMinAlpha(0)
                .setIterations(1)
                .setNumPartitions(1)
                .setNumWords(5);

        word2Vec.train(corpus);

        Collection<String> words = word2Vec.wordsNearest("day", 0, 40);
        System.out.println();
        System.out.println("day(0): " + words);


        /*System.out.println(word2Vec.similarity("day", 0, "year", 0));
        System.out.println(word2Vec.similarity("day", 0, "should", 0));
        System.out.println(word2Vec.similarity("man", 0, "king", 0));
        System.out.println(word2Vec.similarity("man", 0, "you", 0));
        System.out.println(word2Vec.similarity("man", 0, "woman", 0));
*/
        sc.stop();

        long endtime = System.currentTimeMillis();
        long costTime = (endtime - begintime);
        System.out.println("costTime:" + String.valueOf(costTime / 1000.0) + "s");
    }
}
