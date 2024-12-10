package be.kuleuven.pylos.player.student;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGameIF;
import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosPlayerColor;
import be.kuleuven.pylos.game.PylosSphere;
import be.kuleuven.pylos.game.PylosSquare;
import be.kuleuven.pylos.player.PylosPlayer;

public class MLPlayer extends PylosPlayer {
    private final MoveGenerator moveGenerator = new MoveGenerator();
    private final GameStateTransformer gameStateTransformer = new GameStateTransformer();

    private OrtEnvironment env;
    private OrtSession session;
    private OrtEnvironment env2;
    private OrtSession session2;

    private static final String MODEL_PATH = "./pylos-student/src/main/java/be/kuleuven/pylos/player/student/pylos_model_MP.onnx"; // Dit is het model dat move/place acties doet
    private static final String MODEL_PATH2 = "./pylos-student/src/main/java/be/kuleuven/pylos/player/student/pylos_model_RP.onnx"; // Dit is het model dat remove/pass acties doet


    public MLPlayer() {
        try {
            // Create session options and set thread count
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setIntraOpNumThreads(1);

            env = OrtEnvironment.getEnvironment();
            session = env.createSession(MODEL_PATH, sessionOptions);
            env2 = OrtEnvironment.getEnvironment();
            session2 = env2.createSession(MODEL_PATH2, sessionOptions);
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

class Action {
    public ActionType type;
    public PylosSphere sphere;
    public PylosLocation location;

    public Action(ActionType type, PylosSphere sphere, PylosLocation location) {
        this.type = type;
        this.sphere = sphere;
        this.location = location;
    }

    public Action(ActionType type) {
        this.type = type;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Action action) {
            if (this.type == ActionType.PASS || action.type == ActionType.PASS || this.type == ActionType.REMOVE_FIRST ||  action.type == ActionType.REMOVE_FIRST || this.type == ActionType.REMOVE_SECOND || action.type == ActionType.REMOVE_SECOND) {
                return this.type == action.type && (this.type == ActionType.PASS || this.sphere.ID == action.sphere.ID);
            }
            try {
               return this.type == action.type && this.sphere.ID == action.sphere.ID && this.location.X == action.location.X && this.location.Y == action.location.Y && this.location.Z == action.location.Z;
            } catch (Exception e) {
                return false;
            }
        }
        return false;
    }
}

enum ActionType {
    PLACE, MOVE, REMOVE_FIRST, REMOVE_SECOND, PASS
}

class MoveGenerator {
    GameStateTransformer gameStateTransformer = new GameStateTransformer();

    public List<Action> getPlaceMoves(PylosBoard board, PylosPlayerColor player) {
        List<Action> actions = new ArrayList<>();
        if (board.getReservesSize(player) == 0) {
            return actions;
        }
        
        PylosSphere reserveSphere = board.getReserve(player);
        for (PylosLocation location : board.getLocations()) {
            if (location.isUsable()) {
                if (reserveSphere != null) {
                    actions.add(new Action(ActionType.PLACE, reserveSphere, location));
                }
            }
        }

        return actions;
    }

    public List<Action> getMoveMoves(PylosBoard board, PylosPlayerColor player) {
        List<Action> actions = new ArrayList<>();

        List<PylosLocation> freeLocations = new ArrayList<>();
        for (PylosSquare square: board.getAllSquares()) {
            if (square.getTopLocation().isUsable()) freeLocations.add(square.getTopLocation());
        }

        for (PylosSphere sphere : board.getSpheres(player)) {
            if (!sphere.isReserve() && sphere.canMove() && sphere.getLocation().Z < 3) {
                for (PylosLocation newLocation : freeLocations) {
                    if (newLocation != sphere.getLocation() && newLocation.Z > sphere.getLocation().Z && !sphere.getLocation().isBelow(newLocation)) {
                        actions.add(new Action(ActionType.MOVE, sphere, newLocation));
                    }
                }
            }
        }

        return actions;
    }

    public List<Action> getRemoveMoves(PylosBoard board, PylosPlayerColor player, ActionType actionType) {
        List<Action> actions = new ArrayList<>();

        for (PylosSphere sphere : board.getSpheres(player)) {
            if (sphere.canRemove()) {
                actions.add(new Action(actionType, sphere, null));
            }
        }
        // String boardstr = gameStateTransformer.convertBoardToString(board, player);
        if (actions.isEmpty() && actionType == ActionType.REMOVE_FIRST) {
            System.out.println("no actions possible");
        }

        return actions;
    }

    public List<Action> getMoveOrPlaceMoves(PylosBoard board, PylosPlayerColor player) {
        List<Action> actions = getMoveMoves(board, player);
        actions.addAll(getPlaceMoves(board, player));
        return actions;
    }

    public List<Action> getRemoveOrPassMoves(PylosBoard board, PylosPlayerColor player) {
        List<Action> actions = getRemoveMoves(board, player, ActionType.REMOVE_SECOND);
        actions.add(new Action(ActionType.PASS));
        return actions;
    }
}

class GameStateTransformer {
    private static final int MOVE_L0_START = 0;       // 0-207 (level 0 to 1 or 2)
    private static final int MOVE_L1_START = 208;     // 208-243 (level 1 to 2)
    private static final int PLACE_ACTIONS_START = 244;   // 244-272
    private static final int REMOVE_ACTIONS_START = 273;  // 273-302
    private static final int PASS_ACTION_INDEX = 303;     // 303
    private static final int TOTAL_ACTIONS = 304;

    public String convertBoardToString(PylosBoard board, PylosPlayerColor playerColor) {
        // make a string of the board which consists of 30 digits, each one representing a location on the board, 0 is empty, 1 is player 1, 2 is player 2
        // first 16 z = 0
        // next 8 z = 1
        // next 4 z = 2
        // next 1 z = 3
        String boardString = "";
        for (int z = 0 ; z < 4 ; z++) {
            int size = 4-z;
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    PylosSphere sphere = board.getBoardLocation(x, y, z).getSphere();
                    if (sphere == null) boardString += "0";
                    else {
                        if (sphere.PLAYER_COLOR == playerColor) boardString += "1";
                        else if (sphere.PLAYER_COLOR == playerColor.other()) boardString += "2";
                        else boardString += "0";
                    }
                }
            }
        }
        if (boardString.length() != 30) {
            System.out.println(boardString+" length: "+boardString.length());
            throw new RuntimeException("Board string is not 30 digits long");
        }
        return boardString;
    }

    public String convertActionToString(Action action) {
        char[] actions = new char[TOTAL_ACTIONS];
        Arrays.fill(actions, '0');
        switch (action.type) {
            case MOVE -> {
                int moveIndex = getMoveActionIndex(
                        action.sphere.getLocation(),
                        action.location
                );
                if (moveIndex != -1) {
                    actions[moveIndex] = '1';
                }
            }
                
            case PLACE -> {
                if (action.location.Z != 3) {  // Can't place directly on top
                    actions[PLACE_ACTIONS_START + getLocationIndex(action.location)] = '1';
                }
            }
                
            case REMOVE_FIRST, REMOVE_SECOND -> actions[REMOVE_ACTIONS_START + getLocationIndex(action.sphere.getLocation())] = '1';
            case PASS -> actions[PASS_ACTION_INDEX] = '1';
        }
        
        return new String(actions);
    }

    private int getMoveActionIndex(PylosLocation from, PylosLocation to) {
        int fromZ = from.Z;
        int toZ = to.Z;
        
        // Only level 0 can move to level 1 or 2
        // Only level 1 can move to level 2
        if (fromZ == 0) {
            int fromIndex = getLocationIndexInLevel(from, 0); // 0-15
            if (toZ == 1) {
                // Moving to level 1 (first 9 possible locations per source)
                return MOVE_L0_START + (fromIndex * 13) + getLocationIndexInLevel(to, 1);
            } else if (toZ == 2) {
                // Moving to level 2 (next 4 possible locations per source)
                return MOVE_L0_START + (fromIndex * 13) + 9 + getLocationIndexInLevel(to, 2);
            }
        } else if (fromZ == 1) {
            if (toZ == 2) {
                // Moving from level 1 to level 2
                int fromIndex = getLocationIndexInLevel(from, 1); // 0-8
                return MOVE_L1_START + (fromIndex * 4) + getLocationIndexInLevel(to, 2);
            }
        }
        
        return -1;  // Invalid move
    }
    
    private int getLocationIndexInLevel(PylosLocation location, int level) {
        return switch (level) {
            case 0 -> location.Y * 4 + location.X;
            case 1 -> location.Y * 3 + location.X;
            case 2 -> location.Y * 2 + location.X;
            default -> -1;
        };
    }
        
    private int getLocationIndex(PylosLocation location) {
        int index = 0;
        switch (location.Z) {
            case 0 -> index = location.Y * 4 + location.X; // 4x4 grid
            case 1 -> index = 16 + location.Y * 3 + location.X; // 3x3 grid
            case 2 -> index = 25 + location.Y * 2 + location.X; // 2x2 grid
            case 3 -> index = 29; // single location at the top
        }
        return index;
    }

    public String rotateBoard(String boardStr, int rotations) {
        String rotatedBoard = boardStr;
    
        for (int r = 0; r < rotations; r++) {
            StringBuilder temp = new StringBuilder();
                
            // For each level z
            for (int z = 0; z < 4; z++) {
                int size = 4 - z;
                // Read the board in rotated order
                for (int x = 0; x < size; x++) {
                    for (int y = size - 1; y >= 0; y--) {
                        // Calculate the index in the original string
                        int originalIndex = getIndex(x, y, z);
                        temp.append(rotatedBoard.charAt(originalIndex));
                    }
                }
            }
            
            rotatedBoard = temp.toString();
        }
        
        return rotatedBoard;
    }

    private int getIndex(int x, int y, int z) {
        int index = 0;
        // Add all positions from previous levels
        for (int i = 0; i < z; i++) {
            index += (4-i) * (4-i);
        }
        // Add positions from current level
        return index + y * (4-z) + x;
    }

    public String rotateAction(String actionStr, int rotations) {
        // Find which action it is (where is the '1' in the string)
        int actionIndex = actionStr.indexOf('1');
        if (actionIndex == -1 || actionIndex == PASS_ACTION_INDEX) {
            return actionStr; // Pass action or invalid action
        }
    
        char[] rotatedAction = new char[TOTAL_ACTIONS];
        Arrays.fill(rotatedAction, '0');
    
        if (actionIndex >= PLACE_ACTIONS_START && actionIndex < REMOVE_ACTIONS_START) {
            // Handle PLACE action
            int locationIndex = actionIndex - PLACE_ACTIONS_START;
            // Use the same rotation logic as board rotation to get new position
            int rotatedLocationIndex = rotateLocationIndex(locationIndex, rotations);
            rotatedAction[PLACE_ACTIONS_START + rotatedLocationIndex] = '1';
        }
        else if (actionIndex >= REMOVE_ACTIONS_START) {
            // Handle REMOVE action
            int locationIndex = actionIndex - REMOVE_ACTIONS_START;
            // Use the same rotation logic as board rotation to get new position
            int rotatedLocationIndex = rotateLocationIndex(locationIndex, rotations);
            rotatedAction[REMOVE_ACTIONS_START + rotatedLocationIndex] = '1';
        }
        else {
            // Handle MOVE action
            // Decode the move indices (from and to positions)
            int[] fromTo = decodeMoveAction(actionIndex);
            // Rotate both positions
            int rotatedFrom = rotateLocationIndex(fromTo[0], rotations);
            int rotatedTo = rotateLocationIndex(fromTo[1], rotations);
            // Encode back to move action
            int rotatedMoveIndex = encodeMoveAction(rotatedFrom, rotatedTo);
            rotatedAction[rotatedMoveIndex] = '1';
        }
    
        return new String(rotatedAction);
    }
    
    private int rotateLocationIndex(int index, int rotations) {
        // Convert index to x,y,z coordinates
        int z = 0;
        while (index >= (4-z)*(4-z)) {
            index -= (4-z)*(4-z);
            z++;
        }
        int size = 4-z;
        int y = index / size;
        int x = index % size;
        
        // Apply rotation
        for (int r = 0; r < rotations; r++) {
            int newX = size - 1 - y;
            int newY = x;
            x = newX;
            y = newY;
        }
        
        // Convert back to index
        return getIndex(x, y, z);
    }
    
    private int[] decodeMoveAction(int moveIndex) {
        int actionIndex = moveIndex;  // Nieuwe variabele om mee te werken
        
        if (actionIndex < MOVE_L1_START) {
            // Level 0 move
            int fromPos = actionIndex / 13;
            int toPos = actionIndex % 13;
            if (toPos < 9) {
                // Moving to level 1
                return new int[]{fromPos, 16 + toPos};
            } else {
                // Moving to level 2
                return new int[]{fromPos, 25 + (toPos - 9)};
            }
        } else {
            // Level 1 move
            actionIndex -= MOVE_L1_START;
            int fromPos = (actionIndex / 4) + 16;
            int toPos = 25 + (actionIndex % 4);
            return new int[]{fromPos, toPos};
        }
    }
    
    private int encodeMoveAction(int fromPos, int toPos) {
        if (fromPos < 16) {
            // Moving from level 0
            if (toPos >= 25) {
                // To level 2
                return fromPos * 13 + (toPos - 25 + 9);
            } else {
                // To level 1
                return fromPos * 13 + (toPos - 16);
            }
        } else {
            // Moving from level 1 to level 2
            return MOVE_L1_START + ((fromPos - 16) * 4) + (toPos - 25);
        }
    }

    public Action convertStringToAction(String actionStr, PylosBoard board, PylosPlayerColor color, ActionType type) {
        int actionIdx = actionStr.indexOf('1');
        
        if (actionIdx == PASS_ACTION_INDEX) {
            if (type != ActionType.REMOVE_SECOND) {
                return null; // if it has to do move or removefirst, it should not try to pass
            }
            return new Action(ActionType.PASS);
        }
        
        else if (actionIdx >= REMOVE_ACTIONS_START) {
            // Remove action
            int locationIndex = actionIdx - REMOVE_ACTIONS_START;
            PylosLocation location = getLocationFromIndex(board, locationIndex);
            if (location != null && location.getSphere() != null && location.getSphere().PLAYER_COLOR == color) {
                if (type != ActionType.REMOVE_FIRST && type != ActionType.REMOVE_SECOND) {
                    return null; // if it has to do move or place, it should not do remove
                }
                return new Action(type, location.getSphere(), null);
            }
            else if (location != null && location.getSphere() != null && location.getSphere().PLAYER_COLOR != color) {
                return null; // if it is not the right color, it should not do remove
            }
        } 
        
        else if (actionIdx >= PLACE_ACTIONS_START) {
            // Place action
            int locationIndex = actionIdx - PLACE_ACTIONS_START;
            PylosLocation location = getLocationFromIndex(board, locationIndex);
            if (type != ActionType.MOVE) {
                return null; // if it has to do remove or pass, it should not do place
            }
            if (location != null) {
                if (board.getReservesSize(color) == 0) {
                    return null; // if there are no reserves, it should not do place
                }
                return new Action(ActionType.PLACE, board.getReserve(color), location);
            }
        } 
        else {
            // Move action
            int[] fromTo = decodeMoveAction(actionIdx);
            PylosLocation fromLoc = getLocationFromIndex(board, fromTo[0]);
            PylosLocation toLoc = getLocationFromIndex(board, fromTo[1]);
            if (type != ActionType.MOVE) {
                return null; // if it has to do remove or pass, it should not do move
            }
            if (fromLoc != null && fromLoc.getSphere() != null && fromLoc.getSphere().PLAYER_COLOR == color && toLoc != null && toLoc.getSphere() == null && toLoc.isUsable()) {
                return new Action(ActionType.MOVE, fromLoc.getSphere(), toLoc);
            }
        }
        
        return null;
    }

    private PylosLocation getLocationFromIndex(PylosBoard board, int index) {
        int z = 0;
        while (index >= (4-z)*(4-z)) {
            index -= (4-z)*(4-z);
            z++;
        }
        int size = 4-z;
        int y = index / size;
        int x = index % size;
        
        return board.getBoardLocation(x, y, z);
    }

}