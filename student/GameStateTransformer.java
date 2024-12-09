package be.kuleuven.pylos.player.student;

import java.util.Arrays;

import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosPlayerColor;
import be.kuleuven.pylos.game.PylosSphere;

public class GameStateTransformer {
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