package be.kuleuven.pylos.player.student;

import java.util.ArrayList;
import java.util.List;

import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosPlayerColor;
import be.kuleuven.pylos.game.PylosSphere;
import be.kuleuven.pylos.game.PylosSquare;


public class MoveGenerator {
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
        String boardstr = gameStateTransformer.convertBoardToString(board, player);
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

