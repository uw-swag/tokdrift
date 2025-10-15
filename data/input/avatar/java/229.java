import java.io.*;
import java.util.*;

public class SampleSolution {
    public static final int BUFFERSIZE = 512000;
    private static PrintWriter out =
            new PrintWriter(new BufferedOutputStream(System.out, BUFFERSIZE));

    public static void solve() throws IOException {
        long position = in.nextLong();
        long nrstSeqStartIndx = 1;
        while (getValueAtIndex(nrstSeqStartIndx * 2) < position) {
            nrstSeqStartIndx *= 2;
        }
        while (getValueAtIndex(nrstSeqStartIndx + 1) <= position) nrstSeqStartIndx++;
        long startIndex = getValueAtIndex(nrstSeqStartIndx);
        out.println((position - startIndex) + 1);
    }

    public static long getValueAtIndex(long index) {
        return 1 + ((index - 1) * index / 2);
    }

    public static void main(String args[]) throws Exception {
        in.init(System.in);
        solve();
        out.close();
    }

    public static class in {
        static BufferedReader reader;
        static StringTokenizer tokenizer;

        static void init(InputStream input) {
            reader = new BufferedReader(new InputStreamReader(input), BUFFERSIZE);
            tokenizer = new StringTokenizer("");
        }

        static String next() throws IOException {
            while (!tokenizer.hasMoreTokens()) tokenizer = new StringTokenizer(reader.readLine());
            return tokenizer.nextToken();
        }

        static int nextInt() throws IOException {
            return Integer.parseInt(next());
        }

        static double nextDouble() throws IOException {
            return Double.parseDouble(next());
        }

        static long nextLong() throws IOException {
            return Long.parseLong(next());
        }
    }
}
