package be.kuleuven.pylos.player.student;
import be.kuleuven.pylos.battle.BattleMT;
import be.kuleuven.pylos.battle.BattleResult;
import be.kuleuven.pylos.player.PylosPlayerType;

public class BattleMT2 {
    private static GameLogger gameLogger;
    
    public static BattleResult play(PylosPlayerType c1, PylosPlayerType c2, int runs, int nThreads) {
        // Initialize the logger
        gameLogger = new GameLogger("battle_data.txt");
        try {
            BattleResult result = BattleMT.play(c1, c2, runs, nThreads);

            // PylosPlayer p1 = new PylosPlayerMiniMax(4);
            // PylosPlayer p2 = new BenjaminPlayer();
            // BattleResult result = Battle.play(p1, p2, runs);
            
            // Flush any remaining data
            gameLogger.forceFlush();
            return result;
        } catch (Exception e) {
            throw new RuntimeException("Failed to create player instances", e);
        }
    }

    
    
    public static GameLogger getLogger() {
        return gameLogger;
    }
}