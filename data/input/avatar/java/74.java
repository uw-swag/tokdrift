import java.math.BigDecimal;
import java.util.*;
import java.text.DecimalFormat;

public class SampleSolution {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int A = sc.nextInt();
        int B = sc.nextInt();
        int H = sc.nextInt();
        int M = sc.nextInt();
        BigDecimal AA = new BigDecimal(A);
        BigDecimal BB = new BigDecimal(B);
        BigDecimal HH = new BigDecimal(H);
        BigDecimal MM = new BigDecimal(M);
        BigDecimal ans2;
        BigDecimal kaku;
        BigDecimal mkaku;
        BigDecimal hkaku;
        BigDecimal AA2;
        BigDecimal BB2;
        BigDecimal CC;
        BigDecimal DD;
        double dkaku;
        double dans2;
        mkaku = MM.multiply(BigDecimal.valueOf(6));
        hkaku = HH.multiply(BigDecimal.valueOf(30));
        hkaku = hkaku.add(BigDecimal.valueOf((double) M / 2));
        kaku = mkaku.subtract(hkaku);
        dkaku = Math.abs(Math.toRadians(kaku.doubleValue()));
        AA2 = AA.multiply(AA);
        BB2 = BB.multiply(BB);
        CC = AA2.add(BB2);
        DD = BigDecimal.valueOf(Math.cos(dkaku));
        DD = DD.multiply(BigDecimal.valueOf(2));
        DD = DD.multiply(AA);
        DD = DD.multiply(BB);
        ans2 = CC.subtract(DD);
        dans2 = ans2.doubleValue();
        double ans = Math.sqrt(dans2);
        DecimalFormat format = new DecimalFormat("#.#");
        format.setMinimumFractionDigits(20);
        System.out.println(format.format(ans));
    }
}
