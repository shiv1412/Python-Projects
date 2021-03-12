# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:02:47 2020

@author: sharm
"""

# importing Libraries 
import numpy as np
import pandas as pd
import os
import random
import copy
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import to_categorical

# model training class
class Tic_Tac_Toe_model_class:
# parameter initialization class 
    def __init__(self, input_count, output_count, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_count = input_count
        self.output_count = output_count
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(input_count, )))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(output_count, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# dataset training class 
    def training_data(self, dataset):
        input = []
        output = []
        for data in dataset:
            input.append(data[1])
            output.append(data[0])
# reshaping X and y parameters
        X = np.array(input).reshape((-1, self.input_count))
        y = to_categorical(output, num_classes=3)
        # splittng of training and testing data for features and label
        boundary = int(0.8 * len(X))
        X_train = X[:boundary]
        X_test = X[boundary:]
        y_train = y[:boundary]
        y_test = y[boundary:]
        # model fitting
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, batch_size=self.batch_size)
# predicting values class
    def predict(self, data, index):
        return self.model.predict(np.array(data).reshape(-1, self.input_count))[0][index]

# parameter initialization for various game phases 
X_player = 'X'
O_player = 'O'
EMPTY = ' '
X_player_value = -1
O_player_value = 1
blank_value = 0
X_game_current_status = -1
O_game_current_status = 1
game_draw = 0
game_not_ended = 2

# game logic class
class Tic_toc_toe_Game:
# parameter initlization function
    def __init__(self):
        self.reset()
        self.game_training_record = []
# reset function before resetting parameters when game starts
    def reset(self):
        self.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.boardHistory = []
# displaying values on the board  
    def printBoard(self):
        for i in range(len(self.board)):
            print(' ', end='')
            for j in range(len(self.board[i])):
                if X_player_value == self.board[i][j]:
                    print(X_player, end='')
                elif O_player_value == self.board[i][j]:
                    print(O_player, end='')
                elif blank_value == self.board[i][j]:
                    print(EMPTY, end='')
            print(os.linesep)
# result of the game
    def Game_result(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == blank_value:
                    return game_not_ended

        # calculation of rows on board
        for i in range(len(self.board)):
            candidate = self.board[i][0]
            for j in range(len(self.board[i])):
                if candidate != self.board[i][j]:
                    candidate = 0
            if candidate != 0:
                return candidate

        # calculation of columns on board
        for i in range(len(self.board)):
            candidate = self.board[0][i]
            for j in range(len(self.board[i])):
                if candidate != self.board[j][i]:
                    candidate = 0
            if candidate != 0:
                return candidate

        # calculation of diagonal line
        candidate = self.board[0][0]
        for i in range(len(self.board)):
            if candidate != self.board[i][i]:
                candidate = 0
        if candidate != 0:
            return candidate

        # calculation of another diagonal line
        candidate = self.board[0][2]
        for i in range(len(self.board)):
            if candidate != self.board[i][len(self.board[i]) - i - 1]:
                candidate = 0
        if candidate != 0:
            return candidate

        return game_draw

# finding out how many turns are left
    def get_no_of_available_moves(self):
        availableMoves= []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if (self.board[i][j]) == blank_value:
                    availableMoves.append([i, j])
        return availableMoves

    def addToHistory(self, board):
        self.boardHistory.append(board)

    def printHistory(self):
        print(self.boardHistory)

# finidng the state of the player i.e move ,no of moves left 
    def move(self, position, player):
        availableMoves= self.get_no_of_available_moves()
        for i in range(len(availableMoves)):
            if position[0] == availableMoves[i][0] and position[1] == availableMoves[i][1]:
                self.board[position[0]][position[1]] = player
                self.addToHistory(copy.deepcopy(self.board))

# training of neural network
    def train_neural_network(self, neural_network_player, model):
        playerToMove = X_player_value
        while (self.Game_result() == game_not_ended):
            availableMoves= self.get_no_of_available_moves()
            if playerToMove == neural_network_player:
                maxValue = 0
                bestMove = availableMoves[0]
                for availableMove in availableMoves:
                    # get a copy of a board
                    boardCopy = copy.deepcopy(self.board)
                    boardCopy[availableMove[0]][availableMove[1]] = neural_network_player
                    if neural_network_player == X_player_value:
                        value = model.predict(boardCopy, 0)
                    else:
                        value = model.predict(boardCopy, 2)
                    if value > maxValue:
                        maxValue = value
                        bestMove = availableMove
                selectedMove = bestMove
            else:
                selectedMove = availableMoves[random.randrange(0, len(availableMoves))]
            self.move(selectedMove, playerToMove)
            if playerToMove == X_player_value:
                playerToMove = O_player_value
            else:
                playerToMove = X_player_value

    def getgame_training_record(self):
        return self.game_training_record
    

# automation of training process 
    def automated_training(self, playerToMove):
        while (self.Game_result() == game_not_ended):
            availableMoves= self.get_no_of_available_moves()
            selectedMove = availableMoves[random.randrange(0, len(availableMoves))]
            self.move(selectedMove, playerToMove)
            if playerToMove == X_player_value:
                playerToMove = O_player_value
            else:
                playerToMove = X_player_value
        # Get the history and build the training set
        for historyItem in self.boardHistory:
            self.game_training_record.append((self.Game_result(), copy.deepcopy(historyItem)))


# automation of game
    def automate_games(self, playerToMove, count_games):
        playerXWins = 0
        playerOWins = 0
        draws = 0
        for i in range(count_games):
            self.reset()
            self.automated_training(playerToMove)
            if self.Game_result() == X_player_value:
                playerXWins = playerXWins + 1
            elif self.Game_result() == O_player_value:
                playerOWins = playerOWins + 1
            else: draws = draws + 1
        totalWins = playerXWins + playerOWins + draws
        print ('Congrats Player X Wins: ' + str(int(playerXWins * 100/totalWins)) + '%')
        print('Congrats Player O Wins: ' + str(int(playerOWins * 100 / totalWins)) + '%')
        print('Oops this is a Draw: ' + str(int(draws * 100 / totalWins)) + '%')

# automation of the process of training of neural network
    def automated_training_neural_network(self, neural_network_player, count_games, model):
        neural_network_player_winning = 0
        non_neural_network_player_win = 0
        draws = 0
        print ("Neural network player")
        print (neural_network_player)
        for i in range(count_games):
            self.reset()
            self.train_neural_network(neural_network_player, model)
            if self.Game_result() == neural_network_player:
                neural_network_player_winning = neural_network_player_winning + 1
            elif self.Game_result() == game_draw:
                draws = draws + 1
            else: non_neural_network_player_win = non_neural_network_player_win + 1
        totalWins = neural_network_player_winning + non_neural_network_player_win + draws
        print ('Hurray!! Player X Wins: ' + str(int(neural_network_player_winning * 100/totalWins)) + '%')
        print('Hurray Player O Wins: ' + str(int(non_neural_network_player_win * 100 / totalWins)) + '%')
        print('Sorry this is a Draws. Please try next time: ' + str(int(draws * 100 / totalWins)) + '%')

if __name__ == "__main__":
    tic_tac_toe_game = Tic_toc_toe_Game()
    tic_tac_toe_game.automate_games(1, 100)
    Tic_Tac_Toe_model_class = Tic_Tac_Toe_model_class(9, 3, 100, 32)
    Tic_Tac_Toe_model_class.training_data(tic_tac_toe_game.getgame_training_record())
    print ("training Neural Network for  X Player")
    tic_tac_toe_game.automated_training_neural_network(X_player_value, 100, Tic_Tac_Toe_model_class)
    print("training Neural Network for O Player:")
    tic_tac_toe_game.automated_training_neural_network(O_player_value, 100, Tic_Tac_Toe_model_class)
