package haiqing.word2vec;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.util.*;


/**
 * Huffman tree builder
 * @author Adam Gibson
 *
 */
public class Huffman {

    public Huffman(Collection<VocabWord> words) {
        this.words = new ArrayList(words);
    }

    private List<VocabWord> words;
    public final static int MAX_CODE_LENGTH = 40;
    public void build() {
        long[] count = new long[words.size() * 2 + 1];
        int[] binary = new int[words.size() * 2 + 1];
        int[] code = new int[MAX_CODE_LENGTH];
        int[] point = new int[MAX_CODE_LENGTH];
        int[] parentNode = new int[words.size() * 2 + 1];
        int a = 0;

        while (a < words.size()) {
            count[a] = (long) words.get(a).getWordFrequency();
            a++;
        }

        a = words.size();

        while(a < words.size() * 2) {
            count[a] = Integer.MAX_VALUE;
            a++;
        }

        int pos1 = words.size() - 1;
        int pos2 = words.size();

        int min1i;
        int min2i;

        a = 0;
        // Following algorithm constructs the Huffman tree by adding one node at a time
        for (a = 0; a < words.size() - 1; a++) {
            // First, find two smallest nodes 'min1, min2'
            if (pos1 >= 0) {
                if (count[pos1] < count[pos2]) {
                    min1i = pos1;
                    pos1--;
                } else {
                    min1i = pos2;
                    pos2++;
                }
            } else {
                min1i = pos2;
                pos2++;
            }
            if (pos1 >= 0) {
                if (count[pos1] < count[pos2]) {
                    min2i = pos1;
                    pos1--;
                } else {
                    min2i = pos2;
                    pos2++;
                }
            } else {
                min2i = pos2;
                pos2++;
            }

            count[words.size() + a] = count[min1i] + count[min2i];
            parentNode[min1i] = words.size() + a;
            parentNode[min2i] = words.size() + a;
            binary[min2i] = 1;
        }
        // Now assign binary code to each vocabulary word
        int i ;
        int b;
        // Now assign binary code to each vocabulary word
        for (a = 0; a < words.size(); a++) {
            b = a;
            i = 0;
            do {
                code[i] = binary[b];
                point[i] = b;
                i++;
                b = parentNode[b];

            } while(b != words.size() * 2 - 2 && i < 39);


            words.get(a).setCodeLength(i);
            words.get(a).getPoints().add(words.size() - 2);
            words.get(a).getPoints().set(0, words.size() - 2);

            for (b = 0; b < i; b++) {
                words.get(a).getCodes().set(i - b - 1,code[b]);
                words.get(a).getPoints().set(i - b,point[b] - words.size());
            }
        }


    }

}
