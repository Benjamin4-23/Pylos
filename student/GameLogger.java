package be.kuleuven.pylos.player.student;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class GameLogger {
    private static final int BUFFER_THRESHOLD = 400; // Adjust this value as needed
    private final String filename;
    private final ConcurrentLinkedQueue<String> buffer;
    private final AtomicInteger bufferSize;
    
    public GameLogger(String filename) {
        this.filename = filename;
        this.buffer = new ConcurrentLinkedQueue<>();
        this.bufferSize = new AtomicInteger(0);
    }
    
    public void log(String currentState, String action, double reward) {
        String line = currentState + "," + action + "," + reward;
        buffer.offer(line);
        
        if (bufferSize.incrementAndGet() >= BUFFER_THRESHOLD) {
            synchronized(this) {
                if (bufferSize.get() >= BUFFER_THRESHOLD) {
                    flushBuffer();
                }
            }
        }
    }
    
    private void flushBuffer() {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename, true))) {
            String line;
            while ((line = buffer.poll()) != null) {
                writer.write(line);
                writer.newLine();
                bufferSize.decrementAndGet();
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }
    
    public void forceFlush() {
        synchronized(this) {
            flushBuffer();
        }
    }
}