package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosSphere;

public class Action {
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
