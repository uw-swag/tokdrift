import java.io.*;

public class SampleSolution {
    public static void main(String[] args) throws IOException {
        BufferedReader buf = new BufferedReader(new InputStreamReader(System.in));
        String[] inp = buf.readLine().split(" ");
        int n = Integer.parseInt(inp[0]);
        int m = Integer.parseInt(inp[1]);
        String[] ans = new String[n];
        for (int i = 0; i < n; i++) {
            String str = buf.readLine();
            StringBuilder temp = new StringBuilder("");
            for (int j = 0; j < m; j++)
                if (str.charAt(j) == '-') temp.append("-");
                else {
                    if ((i + j) % 2 == 1) temp.append("W");
                    else temp.append("B");
                }
            ans[i] = temp.toString();
        }
        for (int i = 0; i < n; i++) {
            System.out.println(ans[i]);
        }
    }
}
