import java.util.Scanner;
import java.util.stream.IntStream;
import java.text.DecimalFormat;

public class SampleSolution {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int N = sc.nextInt();
        double[] x = new double[N];
        String[] u = new String[N];
        for (int i = 0; i < N; ++i) {
            x[i] = sc.nextDouble();
            u[i] = sc.next();
        }
        DecimalFormat format = new DecimalFormat("#.#");
        format.setMinimumFractionDigits(4);
        System.out.println(format.format(solve(x, u)));
        sc.close();
    }

    static double solve(double[] x, String[] u) {
        return IntStream.range(0, x.length)
                .mapToDouble(i -> x[i] * (u[i].equals("JPY") ? 1 : 380000))
                .sum();
    }
}
