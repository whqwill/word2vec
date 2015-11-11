package haiqing.word2vec; /**
 * Created by hwang on 05.11.15.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

public class testSpark {
    public static void main(String[] args) throws Exception {
        SparkConf sparkConf = new SparkConf().setAppName("sparktest");
        if (!sparkConf.contains("master")) {
            sparkConf.setMaster("local[*]");
        }
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> lines = sc.textFile("raw_sentences.txt");
        JavaRDD<Integer> lineLengths = lines.map(new Function<String, Integer>() {
            public Integer call(String s) {
                return s.length(); }
        });
        int totalLength = lineLengths.reduce(new Function2<Integer, Integer, Integer>() {
            public Integer call(Integer a, Integer b) { return a + b; }
        });
        System.out.println(totalLength);
    }
}
