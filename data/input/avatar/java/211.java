import java.io.*;
import java.util.*;

public class SampleSolution {
    public static void main(String[] args) throws Exception {
        FastReader in = new FastReader();
        int n = in.nextInt();
        TreeSet<Integer> left = new TreeSet<>();
        int[] answer = new int[n];
        for (int i = 0; i < n; i++) {
            left.add(i);
        }
        int q = in.nextInt();
        while (q-- > 0) {
            int l = in.nextInt() - 1;
            int r = in.nextInt() - 1;
            int win = in.nextInt();
            while (left.ceiling(l) != null && left.ceiling(l) <= r) {
                int curr = left.ceiling(l);
                answer[curr] = win;
                left.remove(curr);
            }
            answer[win - 1] = 0;
            left.add(win - 1);
        }
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < n; i++) {
            ans.append(answer[i] + " ");
        }
        System.out.println(ans);
    }

    static class FastReader {
        StringTokenizer st;
        BufferedReader br;

        public FastReader() {
            br = new BufferedReader(new InputStreamReader(System.in));
        }

        String next() {
            while (st == null || !st.hasMoreElements()) {
                try {
                    st = new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }

        int nextInt() {
            return Integer.parseInt(next());
        }

        long nextLong() {
            return Long.parseLong(next());
        }

        double nextDouble() {
            return Double.parseDouble(next());
        }

        String nextLine() {
            String s = "";
            while (st == null || st.hasMoreElements()) {
                try {
                    s = br.readLine();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return s;
        }
    }
}
