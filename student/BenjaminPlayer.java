package be.kuleuven.pylos.player.student;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGameIF;
import be.kuleuven.pylos.game.PylosGameSimulator;
import be.kuleuven.pylos.game.PylosGameState;
import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosPlayerColor;
import be.kuleuven.pylos.game.PylosSquare;
import be.kuleuven.pylos.player.PylosPlayer;


public class BenjaminPlayer extends PylosPlayer {
    private static final int MAX_DEPTH = 7;
    private final MoveGenerator moveGenerator = new MoveGenerator();
    private final GameStateTransformer gameStateTransformer = new GameStateTransformer();

    public BenjaminPlayer() {}

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        // current state
        String currentState = gameStateTransformer.convertBoardToString(board, this.PLAYER_COLOR);
        // current evaluation score 
        int currentEvaluation = newEvaluation(board);


        PylosGameSimulator simulator = new PylosGameSimulator(game.getState(), this.PLAYER_COLOR, board);
        PylosPlayerColor myColor = this.PLAYER_COLOR;
        Action bestMove = findBestMove(board, simulator, myColor);
        // haal move string hier op
        String bestMoveStr = gameStateTransformer.convertActionToString(bestMove);
        if (bestMove.type == ActionType.MOVE) game.moveSphere(bestMove.sphere, bestMove.location);
        else game.moveSphere(board.getReserve(myColor), bestMove.location);


        // new evaluation score
        int newEvaluation = newEvaluation(board);
        // reward heeft hij squares van tegenstander geblockt, heeft hij blockable squares gemaakt, heeft hij squares volledig gemaakt
        double reward = (newEvaluation - currentEvaluation + 1) != 0 ? (newEvaluation - currentEvaluation + 1) : 0.1;
        
        
        // write to file
        writeToFile(currentState, bestMoveStr, reward);
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        // current state
        String currentState = gameStateTransformer.convertBoardToString(board, this.PLAYER_COLOR);
        
        PylosGameSimulator simulator = new PylosGameSimulator(game.getState(), this.PLAYER_COLOR, board);
        PylosPlayerColor myColor = this.PLAYER_COLOR;
        Action bestMove = findBestMove(board, simulator, myColor);

        // haal move string hier op
        String bestMoveStr = gameStateTransformer.convertActionToString(bestMove);

        game.removeSphere(bestMove.sphere);

        // reward is altijd 1 omdat hij 1 reserve meer heeft
        double reward = 2;
        // write to file
        writeToFile(currentState, bestMoveStr, reward);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        // current state
        String currentState = gameStateTransformer.convertBoardToString(board, this.PLAYER_COLOR);
        
        PylosGameSimulator simulator = new PylosGameSimulator(game.getState(), this.PLAYER_COLOR, board);
        PylosPlayerColor myColor = this.PLAYER_COLOR;
        Action bestMove = findBestMove(board, simulator, myColor);

        // haal move string hier op
        String bestMoveStr = gameStateTransformer.convertActionToString(bestMove);

        if (bestMove.type == ActionType.REMOVE_SECOND) game.removeSphere(bestMove.sphere);
        else game.pass();

        // reward bij remove is 1, bij pass is -1 want hij laat blockable square achter
        double reward = bestMove.type == ActionType.REMOVE_SECOND ? 2 : -1;
        // write to file
        writeToFile(currentState, bestMoveStr, reward);
    }

    private void writeToFile(String currentState, String actionStr, double reward) {
        GameLogger logger = BattleMT2.getLogger();
        if (logger != null) {
            logger.log(currentState, actionStr, reward);

            for (int i = 1; i <= 3; i++) {
                String rotatedCurrentState = gameStateTransformer.rotateBoard(currentState, i);
                String rotatedActionStr = gameStateTransformer.rotateAction(actionStr, i);
                // String rotatedNewState = gameStateTransformer.rotateBoard(newState, i);
                logger.log(rotatedCurrentState, rotatedActionStr, reward);
            }
        }
    }

    private int newEvaluation(PylosBoard board) {
        // count reserve spheres
        int reserveSpheres = board.getReservesSize(this.PLAYER_COLOR);
    
        // count blocking squares
        int currentBlockingScore = blockingSquares(board,this.PLAYER_COLOR);
    
        // count blockable squares
        int currentBlockableSquares = blockableSquares(board, this.PLAYER_COLOR) * -1;
    
        // made complete square
        int completeSquares = completedSquares(board, this.PLAYER_COLOR);

        return reserveSpheres + currentBlockingScore + currentBlockableSquares + completeSquares;
    }

    private Action findBestMove(PylosBoard board, PylosGameSimulator simulator, PylosPlayerColor color) {
        Action bestAction = null;
        int bestValue = Integer.MIN_VALUE;
        List<Action> actions = getPossibleActions(board, simulator, color);

        for (Action action : actions) {
            PylosGameState prevState = simulator.getState();
            PylosPlayerColor prevColor = simulator.getColor();
            PylosLocation prevLocation = action.sphere != null ? action.sphere.getLocation() : null;

            applyAction(simulator, action);
            int value = minimax(board, simulator, color, MAX_DEPTH - 1, Integer.MIN_VALUE, Integer.MAX_VALUE);
            undoAction(simulator, action, prevState, prevColor, prevLocation);

            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
             }
        }

        return bestAction;
    }

    private int minimax(PylosBoard board, PylosGameSimulator simulator, PylosPlayerColor color, int depth, int alpha, int beta) {
        if (depth == 0 || simulator.getState() == PylosGameState.COMPLETED || simulator.getState() == PylosGameState.DRAW) {
            return evaluate(board, color);
        }

        PylosPlayerColor currentColor = simulator.getColor();
        List<Action> actions = getPossibleActions(board, simulator, currentColor);

        int maxEval = Integer.MIN_VALUE;
        int minEval = Integer.MAX_VALUE;
        for (Action action : actions) {
            PylosGameState prevState = simulator.getState();
            PylosPlayerColor prevColor = simulator.getColor();
            PylosLocation prevLocation = action.sphere != null ? action.sphere.getLocation() : null;

            applyAction(simulator, action);
            int eval = minimax(board, simulator, color, depth - 1, alpha, beta);
            undoAction(simulator, action, prevState, prevColor, prevLocation);


            if (currentColor == color) {
                maxEval = Math.max(maxEval, eval);
                alpha = Math.max(alpha, eval);
            }
            else {
                minEval = Math.min(minEval, eval);
                beta = Math.min(beta, eval);
            }
            if (beta <= alpha) {
                break;
            }
        }
        if (currentColor == color) return maxEval;
        else return minEval;

    }

    private int evaluate(PylosBoard board, PylosPlayerColor color) {
        // Beste score met deze combinatie. 1000 games: 91 % gewonnen tegen minimax4 in 0.97 s/game
        int score = 0;
        score += (board.getReservesSize(color) - board.getReservesSize(color.other())) ;
        //score += (completedSquares(board,color) - completedSquares(board, color.other())) * 2;
        score += blockingSquares(board,color) - blockingSquares(board, color.other());

        // End of game reward:
        //if (simulator.getState() == PylosGameState.COMPLETED && simulator.getWinner() == color) score += 2;

        return score;
    }

    private int completedSquares(PylosBoard board, PylosPlayerColor color) {
        int squareCount = 0;
        for (PylosSquare square: board.getAllSquares()) {
            int index = 0;
            for (PylosLocation location : square.getLocations()) {
                if (!location.isUsed() || location.getSphere().PLAYER_COLOR != color) break;
                index++;
            }
            if (index == 4) squareCount++;
        }
        return squareCount; 
    }
    private int blockingSquares(PylosBoard board, PylosPlayerColor color) {
        int blockingCount = 0;
        for (PylosSquare square: board.getAllSquares()) {
            int current_color_count = 0;
            int other_color_count = 0;
            int index = 0;

            for (PylosLocation location : square.getLocations()) {
                if (!location.isUsed()) break;

                if (location.getSphere().PLAYER_COLOR == color)  current_color_count++;
                else if (location.getSphere().PLAYER_COLOR != color) other_color_count++;
                index++;
            }
            if (index == 4 && current_color_count == 1 && other_color_count == 3)  blockingCount++;
        }
        return blockingCount;
    }
    private int blockableSquares(PylosBoard board, PylosPlayerColor color) {
        int blockableCount = 0;
        for (PylosSquare square: board.getAllSquares()) {
            int index = 0;
            int sphereCount = 0;
            for (PylosLocation location : square.getLocations()) {
                if (!location.isUsed()) continue;
                if (location.getSphere().PLAYER_COLOR == color) sphereCount++;
                index++;
            }
            if (index == 3 && sphereCount == 3) blockableCount++; // 3 spheres, 3 of this player, 1 empty
        }
        return blockableCount;
    }

    private List<Action> getPossibleActions(PylosBoard board, PylosGameSimulator simulator, PylosPlayerColor color) { // haald zelfde acties op, kijkt naar board ipv simulator?
        List<Action> actions = new ArrayList<>();
        PylosGameState state = simulator.getState();
        switch (state) {
            case MOVE -> actions.addAll(moveGenerator.getMoveOrPlaceMoves(board, color));
            case REMOVE_FIRST -> actions.addAll(moveGenerator.getRemoveMoves(board, color, ActionType.REMOVE_FIRST));
            case REMOVE_SECOND -> actions.addAll(moveGenerator.getRemoveOrPassMoves(board, color));
            default -> {}
        }
        Collections.shuffle(actions, this.getRandom());
        return actions;
    }

    private void applyAction(PylosGameSimulator simulator, Action action) {
        switch (action.type) {
            case PLACE, MOVE -> simulator.moveSphere(action.sphere, action.location);
            case REMOVE_FIRST, REMOVE_SECOND -> simulator.removeSphere(action.sphere);
            case PASS -> simulator.pass();
        }
    }

    private void undoAction(PylosGameSimulator simulator, Action action, PylosGameState prevState, PylosPlayerColor prevColor, PylosLocation prevLocation) {
        switch (action.type) {
            case PLACE -> simulator.undoAddSphere(action.sphere, prevState, prevColor);
            case MOVE -> simulator.undoMoveSphere(action.sphere, prevLocation, prevState, prevColor);
            case REMOVE_FIRST -> simulator.undoRemoveFirstSphere(action.sphere, prevLocation, prevState, prevColor);
            case REMOVE_SECOND -> simulator.undoRemoveSecondSphere(action.sphere, prevLocation, prevState, prevColor);
            case PASS -> simulator.undoPass(prevState, prevColor);
        }
    }
}
