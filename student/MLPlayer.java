package be.kuleuven.pylos.player.student;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGameIF;
import be.kuleuven.pylos.player.PylosPlayer;

public class MLPlayer extends PylosPlayer {
    private final MoveGenerator moveGenerator = new MoveGenerator();
    private final GameStateTransformer gameStateTransformer = new GameStateTransformer();

    private OrtEnvironment env;
    private OrtSession session;
    private OrtEnvironment env2;
    private OrtSession session2;

    private static final String MODEL_PATH = "./pylos-student/src/main/java/be/kuleuven/pylos/player/student/AA_models/pylos_model_MP.onnx"; //"./AA_models/pylos_model_MP.onnx";
    private static final String MODEL_PATH2 = "./pylos-student/src/main/java/be/kuleuven/pylos/player/student/AA_models/pylos_model_RP.onnx";


    public MLPlayer() {
        try {
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(MODEL_PATH);
            env2 = OrtEnvironment.getEnvironment(); // zelfde env gebruiken voor remove pass model
            session2 = env2.createSession(MODEL_PATH2);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        try {
            float[][] qValues = getPrediction(board, false);
            List<Action> validMoves = moveGenerator.getMoveOrPlaceMoves(board, this.PLAYER_COLOR);
            Action bestAction = findBestValidAction(qValues[0], validMoves, board, ActionType.MOVE);
            if (bestAction.type == ActionType.MOVE) game.moveSphere(bestAction.sphere, bestAction.location);
            else game.moveSphere(board.getReserve(this.PLAYER_COLOR), bestAction.location);
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error in doMove");
            executeFirstValidMove(game, board, moveGenerator.getMoveOrPlaceMoves(board, this.PLAYER_COLOR));
        }
    }


    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {

        try {
            float[][] qValues = getPrediction(board, true);
            List<Action> validMoves = moveGenerator.getRemoveMoves(board, this.PLAYER_COLOR, ActionType.REMOVE_FIRST);
            Action bestAction = findBestValidAction(qValues[0], validMoves, board, ActionType.REMOVE_FIRST);
            game.removeSphere(bestAction.sphere);
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error in doMove");
            executeFirstValidMove(game, board, moveGenerator.getRemoveMoves(board, this.PLAYER_COLOR, ActionType.REMOVE_FIRST));
        }
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        try {
            float[][] qValues = getPrediction(board, true);
            List<Action> validMoves = moveGenerator.getRemoveOrPassMoves(board, this.PLAYER_COLOR);
            Action bestAction = findBestValidAction(qValues[0], validMoves, board, ActionType.REMOVE_SECOND);
            if (bestAction.type == ActionType.REMOVE_SECOND) game.removeSphere(bestAction.sphere);
            else game.pass();
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error in doMove");
            executeFirstValidMove(game, board, moveGenerator.getRemoveOrPassMoves(board, this.PLAYER_COLOR));
        }
    }

    
    private void executeFirstValidMove(PylosGameIF game, PylosBoard board, List<Action> validMoves) {
        if (!validMoves.isEmpty()) {
            Action fallbackAction = validMoves.get(0);
            switch (fallbackAction.type) {
                case MOVE -> game.moveSphere(fallbackAction.sphere, fallbackAction.location);
                case PLACE -> game.moveSphere(board.getReserve(this.PLAYER_COLOR), fallbackAction.location);
                case REMOVE_FIRST, REMOVE_SECOND -> game.removeSphere(fallbackAction.sphere);
                case PASS -> game.pass();
            }
        }
    }
    private float[][] getPrediction(PylosBoard board, boolean removePass) throws OrtException {
        String currentState = gameStateTransformer.convertBoardToString(board, this.PLAYER_COLOR);
        float[] stateArray = new float[30];
        for (int i = 0; i < currentState.length(); i++) {
            char c = currentState.charAt(i);
            switch (c) {
                case '0':
                    stateArray[i] = 0.0f;
                    break;
                case '1':
                    stateArray[i] = 1.0f;
                    break;
                case '2':
                    stateArray[i] = 2.0f;
                    break;
                default:
                    System.out.println("Error in getPrediction: invalid character in state string");
                    break;
            }
        }
        OrtSession sessionToUse = removePass ? session2 : session;
        OrtEnvironment envToUse = removePass ? env2 : env;

        OnnxTensor inputTensor = OnnxTensor.createTensor(envToUse, 
            FloatBuffer.wrap(stateArray), new long[]{1, 30});

        OrtSession.Result result = sessionToUse.run(
            Collections.singletonMap("input", inputTensor));

        return (float[][]) result.get(0).getValue();
    }
    private Action findBestValidAction(float[] qValues, List<Action> validMoves, PylosBoard board, ActionType type) {
        if (validMoves.isEmpty()) {
            return null;
        }
        Integer[] indices = new Integer[qValues.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(qValues[b], qValues[a]));


       


        // System.out.println("--------------------------------- geen move boven threshold");
        // Try every possible action in order of decreasing Q-value
        for (int idx : indices) {
            // doe niet altijd de beste
            String actionStr = formatAction(idx);
            Action candidateAction = gameStateTransformer.convertStringToAction(actionStr, board, this.PLAYER_COLOR, type);
            if (candidateAction != null) {
                for (Action validMove : validMoves) {
                    if (validMove.equals(candidateAction)) {
                        // System.out.println("move gedaan met qvalue: " + qValues[idx]);
                        return validMove;
                    }
                }
            }
        }
        return validMoves.get(0);
    }  
    private String formatAction(int actionIdx) {
        char[] actionStr = new char[304];
        Arrays.fill(actionStr, '0');
        actionStr[actionIdx] = '1';
        return new String(actionStr);
    }

    
}

