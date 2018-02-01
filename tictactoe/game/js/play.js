"use strict";

var game;
var ais;
var aiDelay = 1000;
var paused = false;

$(function (){

    var $board = $('#board');
    var $pauseButton = $('#pause');
    var $stepButton = $('#step');
    var $undoButton = $('#undo');
    var $restartButton = $('#restart');

    function setPaused(p) {
        if (p != paused) {
            paused = p;
            update();
        }
    }

    function makeAiMove() {

    }

    function undo() {

    }

    function restart() {

    }

    function update() {
        $pauseButton.val($pauseButton.data(paused ? 'paused' : 'unpaused'));
    }
    $board.mousemove(function (event) {

    });

    $board.mouseleave(function (event) {

    });

    $board.click(function (event) {

    });

    $pauseButton.click(function (event) {
        setPaused(!paused);
    });

    $stepButton.click(function (event) {
        makeAiMove();
    });

    $undoButton.click(function (event) {
        undo();
    });

    $restartButton.click(function (event) {
        restart();
    });
});