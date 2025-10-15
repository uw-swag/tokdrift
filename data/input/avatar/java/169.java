import java.util.Scanner;
import java.text.DecimalFormat;

public class SampleSolution {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int N = scanner.nextInt();
        int[][] pos = new int[N][2];
        for (int i = 0; i < N; i++) {
            pos[i][0] = scanner.nextInt();
            pos[i][1] = scanner.nextInt();
        }
        double sum = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                sum += dist(pos[i][0], pos[i][1], pos[j][0], pos[j][1]);
            }
        }
        DecimalFormat format = new DecimalFormat("#.#");
        format.setMinimumFractionDigits(10);
        System.out.println(format.format(sum / N));
    }

    private static double dist(int x1, int y1, int x2, int y2) {
        return Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    }
}
