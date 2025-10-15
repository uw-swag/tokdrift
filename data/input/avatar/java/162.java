import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;

public class SampleSolution {
    public static void main(String[] args) throws Exception {
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            double input = Double.parseDouble(br.readLine());
            double countOdd = Math.round(input / 2);
            BigDecimal result = new BigDecimal(countOdd / input);
            result.setScale(10, RoundingMode.HALF_UP);
            DecimalFormat format = new DecimalFormat("#.#");
            format.setMinimumFractionDigits(10);
            System.out.println(format.format(result));
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            System.exit(0);
        } catch (final Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }
}
