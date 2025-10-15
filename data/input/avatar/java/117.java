import java.util.Collections;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Scanner;

class Job implements Comparable<Job> {
    int a;
    int b;

    Job(int a, int b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public int compareTo(Job otherJob) {
        if (otherJob.a == this.a) return this.b - otherJob.b;
        else return this.a - otherJob.a;
    }
}

public class SampleSolution {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String[] line = sc.nextLine().split(" ");
        int N = Integer.parseInt(line[0]);
        int M = Integer.parseInt(line[1]);
        Queue<Job> q = new PriorityQueue<>();
        for (int i = 0; i < N; i++) {
            line = sc.nextLine().split(" ");
            q.add(new Job(Integer.parseInt(line[0]), Integer.parseInt(line[1])));
        }
        int cnt = 0;
        Queue<Integer> jobQ = new PriorityQueue<>(Collections.reverseOrder());
        for (int i = 1; i <= M; i++) {
            while (!q.isEmpty()) {
                Job job = q.peek();
                if (job.a <= i) {
                    jobQ.add(q.poll().b);
                } else break;
            }
            if (!jobQ.isEmpty()) cnt += jobQ.poll();
        }
        System.out.println(cnt);
    }
}
