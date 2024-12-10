package be.kuleuven.pylos.main;

import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import be.kuleuven.pylos.battle.Battle;
import be.kuleuven.pylos.battle.BattleMT;
import be.kuleuven.pylos.battle.BattleResult;
import be.kuleuven.pylos.battle.RoundRobin;
import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGame;
import be.kuleuven.pylos.game.PylosGameObserver;
import be.kuleuven.pylos.player.PylosPlayer;
import be.kuleuven.pylos.player.PylosPlayerObserver;
import be.kuleuven.pylos.player.PylosPlayerType;
import be.kuleuven.pylos.player.codes.PlayerFactoryCodes;
import be.kuleuven.pylos.player.codes.PylosPlayerBestFit;
import be.kuleuven.pylos.player.codes.PylosPlayerMiniMax;
import be.kuleuven.pylos.player.student.MLPlayer;

public class PylosMain {

    public static void main(String[] args) {
        //startSingleGame();
        startBattle();
        // startBattleMultithreaded();
        // startRoundRobinTournament();
    }

    public static void startSingleGame() {

        Random random = new Random(0);

        PylosPlayer playerLight = new PylosPlayerBestFit();
        PylosPlayer playerDark = new PylosPlayerMiniMax(2);

        PylosBoard pylosBoard = new PylosBoard();
        PylosGame pylosGame = new PylosGame(pylosBoard, playerLight, playerDark, random, PylosGameObserver.CONSOLE_GAME_OBSERVER, PylosPlayerObserver.NONE);

        pylosGame.play();
    }

    public static void startBattle() {
        int nRuns = 100;

        PylosPlayerType p1 = new PylosPlayerType("Machine Learning Player") {
            @Override
            public PylosPlayer create() {
                return new MLPlayer();
            }
        };

        PylosPlayerType p2 = new PylosPlayerType("Minimax 6") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(6);
            }
        };

        Battle.play(p1, p2, nRuns);
    }

    public static void startBattleMultithreaded() {
        //Please refrain from using Collections.shuffle(List<?> list) in your player,
        //as this is not ideal for use across multiple threads.
        //Use Collections.shuffle(List<?> list, Random random) instead, with the Random object from the player (PylosPlayer.getRandom())

        int nRuns = 5000;
        int nThreads = 5;

        PylosPlayerType p1 = new PylosPlayerType("Machine Learning Player 2") {
            @Override
            public PylosPlayer create() {
                return new MLPlayer();
            }
        };
        PylosPlayerType p2 = new PylosPlayerType("Minimax 6") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(6);
            }
        };

        BattleMT.play(p1, p2, nRuns, nThreads);
    }

    public static void startRoundRobinTournament() {
        //Same requirements apply as for startBattleMultithreaded()

        //Create your own PlayerFactory containing all PlayerTypes you want to test
        PlayerFactoryCodes pFactory = new PlayerFactoryCodes();
        //PlayerFactoryStudent pFactory = new PlayerFactoryStudent();

        int nRunsPerCombination = 1000;
        int nThreads = 8;

        Set<RoundRobin.Match> matches = RoundRobin.createTournament(pFactory);

        RoundRobin.play(matches, nRunsPerCombination, nThreads);

        List<BattleResult> results = matches.stream().map(c -> c.battleResult).collect(Collectors.toList());

        RoundRobin.printWinsMatrix(results, pFactory.getTypes());
    }
}
