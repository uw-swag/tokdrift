import java.io.*;
import java.math.*;
import java.util.*;
import java.util.concurrent.*;

class SampleSolution {
    static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    static FastScanner sc = new FastScanner(br);
    static PrintWriter out = new PrintWriter(System.out);
    static Random rnd = new Random();

    public static void main(String args[]) throws Exception {
        int n = sc.nextInt();
        long[] pre = new long[n + 1];

        for (int i = 1; i <= n; i++) {
            pre[i] = pre[i - 1] + sc.nextLong();
        }

        long min = Long.MAX_VALUE;

        for (int i = 1; i < n; i++) {
            long val1 = pre[i], val2 = pre[n] - pre[i];

            min = Math.min(min, Math.abs(val1 - val2));
        }

        out.println(min);
        out.close();
    }
}

class FastScanner {
    BufferedReader in;
    StringTokenizer st;

    public FastScanner(BufferedReader in) {
        this.in = in;
    }

    public String getNextToken() throws Exception {
        while (st == null || !st.hasMoreTokens()) {
            st = new StringTokenizer(in.readLine());
        }
        return st.nextToken();
    }

    public String next() throws Exception {
        return getNextToken().toString();
    }

    public int nextInt() throws Exception {
        return Integer.parseInt(getNextToken());
    }

    public long nextLong() throws Exception {
        return Long.parseLong(getNextToken());
    }

    public double nextDouble() throws Exception {
        return Double.parseDouble(getNextToken());
    }
}
