package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.player.PylosPlayer;
import be.kuleuven.pylos.player.PylosPlayerFactory;
import be.kuleuven.pylos.player.PylosPlayerType;

/**
 * Created by Jan on 20/02/2015.
 */
public class PlayerFactoryStudent extends PylosPlayerFactory {

    public PlayerFactoryStudent() {
        super("Student");
    }

    @Override
    protected void createTypes() {

        /* example */
        add(new PylosPlayerType("Machine Learning Player") {
            @Override
            public PylosPlayer create() {
                return new MLPlayer();
            }
        });

        add(new PylosPlayerType("Student - Random") {
            @Override
            public PylosPlayer create() {
                return new StudentPlayerRandomFit();
            }
        });
    }
}
