import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.*;

public class SampleSolution {
    static final long MOD1 = 1000000007;
    static final long MOD2 = 998244353;
    static long[] tenmod;
    static final long inv9 = modInv(9);

    public static void main(String[] args) {
        PrintWriter out = new PrintWriter(System.out);
        InputReader sc = new InputReader(System.in);
        int N = sc.nextInt();
        int Q = sc.nextInt();
        tenmod = new long[N + 1];
        tenmod[0] = 1;
        for (int j = 1; j <= N; j++) {
            tenmod[j] = (tenmod[j - 1] * 10L) % MOD2;
        }
        S[] dat = new S[N];
        Arrays.setAll(dat, i -> new S(tenmod[N - i - 1], i, i));
        LazySegTree<S, F> lazySegTree =
                new LazySegTree<S, F>(dat, S::op, S.E, S::map, F::composite, F.I);
        for (int j = 0; j < Q; j++) {
            int l = sc.nextInt() - 1;
            int r = sc.nextInt() - 1;
            long D = sc.nextLong();
            lazySegTree.apply(l, r + 1, new F(D, N));
            out.println(lazySegTree.allProd().sum);
        }
        out.flush();
    }

    static class S {
        static final S E = new S(0, Integer.MAX_VALUE / 2, Integer.MIN_VALUE / 2);
        long sum;
        int l;
        int r;

        public S(long sum, int l, int r) {
            super();
            this.sum = sum;
            this.l = l;
            this.r = r;
        }

        public static S op(S s1, S s2) {
            long sum = s1.sum + s2.sum;
            if (sum >= MOD2) {
                sum -= MOD2;
            }
            return new S(sum, Math.min(s1.l, s2.l), Math.max(s1.r, s2.r));
        }

        static S map(F f, S s) {
            long c = (tenmod[(f.N - s.l)] - tenmod[(f.N - s.r - 1)]);
            if (c < 0) {
                c += MOD2;
            }
            return new S(((f.D * c) % MOD2 * inv9) % MOD2, s.l, s.r);
        }
    }

    static class F {
        static final F I = new F(0, 0);
        long D;
        int N;

        public F(long D, int N) {
            super();
            this.D = D;
            this.N = N;
        }

        public static F composite(F f, F g) {
            return new F(f.D, f.N);
        }
    }

    static long modInv(long x) {
        return modPow(x, MOD2 - 2);
    }

    static long modPow(long x, long y) {
        long z = 1;
        while (y > 0) {
            if (y % 2 == 0) {
                x = (x * x) % MOD2;
                y /= 2;
            } else {
                z = (z * x) % MOD2;
                y--;
            }
        }
        return z;
    }

    static class LazySegTree<S, F> {
        final int MAX;

        final int N;
        final int Log;
        final java.util.function.BinaryOperator<S> Op;
        final S E;
        final java.util.function.BiFunction<F, S, S> Mapping;
        final java.util.function.BinaryOperator<F> Composition;
        final F Id;

        final S[] Dat;
        final F[] Laz;

        @SuppressWarnings("unchecked")
        public LazySegTree(
                int n,
                java.util.function.BinaryOperator<S> op,
                S e,
                java.util.function.BiFunction<F, S, S> mapping,
                java.util.function.BinaryOperator<F> composition,
                F id) {
            this.MAX = n;
            int k = 1;
            while (k < n) k <<= 1;
            this.N = k;
            this.Log = Integer.numberOfTrailingZeros(N);
            this.Op = op;
            this.E = e;
            this.Mapping = mapping;
            this.Composition = composition;
            this.Id = id;
            this.Dat = (S[]) new Object[N << 1];
            this.Laz = (F[]) new Object[N];
            java.util.Arrays.fill(Dat, E);
            java.util.Arrays.fill(Laz, Id);
        }

        public LazySegTree(
                S[] dat,
                java.util.function.BinaryOperator<S> op,
                S e,
                java.util.function.BiFunction<F, S, S> mapping,
                java.util.function.BinaryOperator<F> composition,
                F id) {
            this(dat.length, op, e, mapping, composition, id);
            build(dat);
        }

        private void build(S[] dat) {
            int l = dat.length;
            System.arraycopy(dat, 0, Dat, N, l);
            for (int i = N - 1; i > 0; i--) {
                Dat[i] = Op.apply(Dat[i << 1 | 0], Dat[i << 1 | 1]);
            }
        }

        private void push(int k) {
            if (Laz[k] == Id) return;
            int lk = k << 1 | 0, rk = k << 1 | 1;
            Dat[lk] = Mapping.apply(Laz[k], Dat[lk]);
            Dat[rk] = Mapping.apply(Laz[k], Dat[rk]);
            if (lk < N) Laz[lk] = Composition.apply(Laz[k], Laz[lk]);
            if (rk < N) Laz[rk] = Composition.apply(Laz[k], Laz[rk]);
            Laz[k] = Id;
        }

        private void pushTo(int k) {
            for (int i = Log; i > 0; i--) push(k >> i);
        }

        private void pushTo(int lk, int rk) {
            for (int i = Log; i > 0; i--) {
                if (((lk >> i) << i) != lk) push(lk >> i);
                if (((rk >> i) << i) != rk) push(rk >> i);
            }
        }

        private void updateFrom(int k) {
            k >>= 1;
            while (k > 0) {
                Dat[k] = Op.apply(Dat[k << 1 | 0], Dat[k << 1 | 1]);
                k >>= 1;
            }
        }

        private void updateFrom(int lk, int rk) {
            for (int i = 1; i <= Log; i++) {
                if (((lk >> i) << i) != lk) {
                    int lki = lk >> i;
                    Dat[lki] = Op.apply(Dat[lki << 1 | 0], Dat[lki << 1 | 1]);
                }
                if (((rk >> i) << i) != rk) {
                    int rki = (rk - 1) >> i;
                    Dat[rki] = Op.apply(Dat[rki << 1 | 0], Dat[rki << 1 | 1]);
                }
            }
        }

        public void set(int p, S x) {
            exclusiveRangeCheck(p);
            p += N;
            pushTo(p);
            Dat[p] = x;
            updateFrom(p);
        }

        public S get(int p) {
            exclusiveRangeCheck(p);
            p += N;
            pushTo(p);
            return Dat[p];
        }

        public S allProd() {
            return Dat[1];
        }

        public void apply(int p, F f) {
            exclusiveRangeCheck(p);
            p += N;
            pushTo(p);
            Dat[p] = Mapping.apply(f, Dat[p]);
            updateFrom(p);
        }

        public void apply(int l, int r, F f) {
            if (l > r) {
                throw new IllegalArgumentException(String.format("Invalid range: [%d, %d)", l, r));
            }
            inclusiveRangeCheck(l);
            inclusiveRangeCheck(r);
            if (l == r) return;
            l += N;
            r += N;
            pushTo(l, r);
            for (int l2 = l, r2 = r; l2 < r2; ) {
                if ((l2 & 1) == 1) {
                    Dat[l2] = Mapping.apply(f, Dat[l2]);
                    if (l2 < N) Laz[l2] = Composition.apply(f, Laz[l2]);
                    l2++;
                }
                if ((r2 & 1) == 1) {
                    r2--;
                    Dat[r2] = Mapping.apply(f, Dat[r2]);
                    if (r2 < N) Laz[r2] = Composition.apply(f, Laz[r2]);
                }
                l2 >>= 1;
                r2 >>= 1;
            }
            updateFrom(l, r);
        }

        public int maxRight(int l, java.util.function.Predicate<S> g) {
            inclusiveRangeCheck(l);
            if (!g.test(E)) {
                throw new IllegalArgumentException("Identity element must satisfy the condition.");
            }
            if (l == MAX) return MAX;
            l += N;
            pushTo(l);
            S sum = E;
            do {
                l >>= Integer.numberOfTrailingZeros(l);
                if (!g.test(Op.apply(sum, Dat[l]))) {
                    while (l < N) {
                        push(l);
                        l = l << 1;
                        if (g.test(Op.apply(sum, Dat[l]))) {
                            sum = Op.apply(sum, Dat[l]);
                            l++;
                        }
                    }
                    return l - N;
                }
                sum = Op.apply(sum, Dat[l]);
                l++;
            } while ((l & -l) != l);
            return MAX;
        }

        public int minLeft(int r, java.util.function.Predicate<S> g) {
            inclusiveRangeCheck(r);
            if (!g.test(E)) {
                throw new IllegalArgumentException("Identity element must satisfy the condition.");
            }
            if (r == 0) return 0;
            r += N;
            pushTo(r - 1);
            S sum = E;
            do {
                r--;
                while (r > 1 && (r & 1) == 1) r >>= 1;
                if (!g.test(Op.apply(Dat[r], sum))) {
                    while (r < N) {
                        push(r);
                        r = r << 1 | 1;
                        if (g.test(Op.apply(Dat[r], sum))) {
                            sum = Op.apply(Dat[r], sum);
                            r--;
                        }
                    }
                    return r + 1 - N;
                }
                sum = Op.apply(Dat[r], sum);
            } while ((r & -r) != r);
            return 0;
        }

        private void exclusiveRangeCheck(int p) {
            if (p < 0 || p >= MAX) {
                throw new IndexOutOfBoundsException(
                        String.format("Index %d is not in [%d, %d).", p, 0, MAX));
            }
        }

        private void inclusiveRangeCheck(int p) {
            if (p < 0 || p > MAX) {
                throw new IndexOutOfBoundsException(
                        String.format("Index %d is not in [%d, %d].", p, 0, MAX));
            }
        }

        private int indent = 6;

        public void setIndent(int newIndent) {
            this.indent = newIndent;
        }

        @Override
        public String toString() {
            return makeString(1, 0);
        }

        private String makeString(int k, int sp) {
            if (k >= N) return indent(sp) + Dat[k];
            String s = "";
            s += makeString(k << 1 | 1, sp + indent);
            s += "\n";
            s += indent(sp) + Dat[k] + "/" + Laz[k];
            s += "\n";
            s += makeString(k << 1 | 0, sp + indent);
            return s;
        }

        private static String indent(int n) {
            StringBuilder sb = new StringBuilder();
            while (n-- > 0) sb.append(' ');
            return sb.toString();
        }
    }

    static class InputReader {
        private InputStream in;
        private byte[] buffer = new byte[1024];
        private int curbuf;
        private int lenbuf;

        public InputReader(InputStream in) {
            this.in = in;
            this.curbuf = this.lenbuf = 0;
        }

        public boolean hasNextByte() {
            if (curbuf >= lenbuf) {
                curbuf = 0;
                try {
                    lenbuf = in.read(buffer);
                } catch (IOException e) {
                    throw new InputMismatchException();
                }
                if (lenbuf <= 0) return false;
            }
            return true;
        }

        private int readByte() {
            if (hasNextByte()) return buffer[curbuf++];
            else return -1;
        }

        private boolean isSpaceChar(int c) {
            return !(c >= 33 && c <= 126);
        }

        private void skip() {
            while (hasNextByte() && isSpaceChar(buffer[curbuf])) curbuf++;
        }

        public boolean hasNext() {
            skip();
            return hasNextByte();
        }

        public String next() {
            if (!hasNext()) throw new NoSuchElementException();
            StringBuilder sb = new StringBuilder();
            int b = readByte();
            while (!isSpaceChar(b)) {
                sb.appendCodePoint(b);
                b = readByte();
            }
            return sb.toString();
        }

        public int nextInt() {
            if (!hasNext()) throw new NoSuchElementException();
            int c = readByte();
            while (isSpaceChar(c)) c = readByte();
            boolean minus = false;
            if (c == '-') {
                minus = true;
                c = readByte();
            }
            int res = 0;
            do {
                if (c < '0' || c > '9') throw new InputMismatchException();
                res = res * 10 + c - '0';
                c = readByte();
            } while (!isSpaceChar(c));
            return (minus) ? -res : res;
        }

        public long nextLong() {
            if (!hasNext()) throw new NoSuchElementException();
            int c = readByte();
            while (isSpaceChar(c)) c = readByte();
            boolean minus = false;
            if (c == '-') {
                minus = true;
                c = readByte();
            }
            long res = 0;
            do {
                if (c < '0' || c > '9') throw new InputMismatchException();
                res = res * 10 + c - '0';
                c = readByte();
            } while (!isSpaceChar(c));
            return (minus) ? -res : res;
        }

        public double nextDouble() {
            return Double.parseDouble(next());
        }

        public int[] nextIntArray(int n) {
            int[] a = new int[n];
            for (int i = 0; i < n; i++) a[i] = nextInt();
            return a;
        }

        public long[] nextLongArray(int n) {
            long[] a = new long[n];
            for (int i = 0; i < n; i++) a[i] = nextLong();
            return a;
        }

        public char[][] nextCharMap(int n, int m) {
            char[][] map = new char[n][m];
            for (int i = 0; i < n; i++) map[i] = next().toCharArray();
            return map;
        }
    }
}
