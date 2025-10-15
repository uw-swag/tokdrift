import java.io.*;
import java.nio.CharBuffer;
import java.util.NoSuchElementException;

public class SampleSolution {
    public static void main(String[] args) {
        SimpleScanner scanner = new SimpleScanner(System.in);
        PrintWriter writer = new PrintWriter(System.out);
        int r = scanner.nextInt();
        int d = scanner.nextInt();
        long x = scanner.nextLong();
        for (int i = 0; i < 10; ++i) {
            x = r * x - d;
            writer.println(x);
        }
        writer.close();
    }

    private static class SimpleScanner {
        private static final int BUFFER_SIZE = 10240;
        private Readable in;
        private CharBuffer buffer;
        private boolean eof;

        SimpleScanner(InputStream in) {
            this.in = new BufferedReader(new InputStreamReader(in));
            buffer = CharBuffer.allocate(BUFFER_SIZE);
            buffer.limit(0);
            eof = false;
        }

        private char read() {
            if (!buffer.hasRemaining()) {
                buffer.clear();
                int n;
                try {
                    n = in.read(buffer);
                } catch (IOException e) {
                    n = -1;
                }
                if (n <= 0) {
                    eof = true;
                    return '\0';
                }
                buffer.flip();
            }
            return buffer.get();
        }

        void checkEof() {
            if (eof) throw new NoSuchElementException();
        }

        char nextChar() {
            checkEof();
            char b = read();
            checkEof();
            return b;
        }

        String next() {
            char b;
            do {
                b = read();
                checkEof();
            } while (Character.isWhitespace(b));
            StringBuilder sb = new StringBuilder();
            do {
                sb.append(b);
                b = read();
            } while (!eof && !Character.isWhitespace(b));
            return sb.toString();
        }

        int nextInt() {
            return Integer.valueOf(next());
        }

        long nextLong() {
            return Long.valueOf(next());
        }

        double nextDouble() {
            return Double.parseDouble(next());
        }
    }
}
