import java.io.BufferedReader;
import java.io.InputStreamReader;

public class SampleSolution {

    public static void main(String[] args) {

        String[] s = parseLine().split(" ");
        int a = Integer.parseInt(s[0]);
        int b = Integer.parseInt(s[1]);
        if (b % a == 0) {
            System.out.println(a + b);
        } else {
            System.out.println(b - a);
        }
        return;
    }

    private static String parseLine() {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            return reader.readLine();
        } catch (Exception e) {
            return e.getMessage();
        }
    }
}
