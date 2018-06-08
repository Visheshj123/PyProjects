from IPython.display import clear_output
import sys
import random

def display_board(board):
    print('{} | {} | {}'.format(board[7],board[8],board[9]))
    print('{} | {} | {}'.format(board[4],board[5],board[6]))
    print('{} | {} | {}'.format(board[1],board[2],board[3]))
def player_input(userChoice):
    #userChoice = ''
    if userChoice != 'X' or userChoice != 'O':
        userChoice = input("Would you like to be X or O?")

    if userChoice == 'X':
        print("You will be X")
        return userChoice
    else:
        userChoice == 'O'
        print("You will be O")
        return userChoice
def place_marker(board, marker, position):
        if marker == 'O':
            marker == 'X'
        else: marker == 'O'
        print(marker)
        board[position] = marker
def win_check(board, mark):
        if board[7:10] == [mark]*3 or board[4:7] == [mark]*3 or board[1:4] == [mark]*3:
            return True
        elif board[3:8:2] == [mark]*3 or board[1:10:4] == [mark]*3:
            return True
        elif board[1:8:3] == [mark]*3 or board[2:9:3] == [mark]*3 or board[3:10:3] == [mark]*3:
            return True
        else: return False
def choose_first(player1, player2,board):
    choice = random.randint(0,1)
    if choice == 1:
        print("playerOne goes first")
        playerChoice(board, player1)
        return player1
    else:
        print("PlayerTwo goes first")
        playerChoice(board, player2)
        return player2
def spaceCheck(board,position):
       return board[position] == ''
def fullBoardCheck(board):
    return board.count('X') + board.count('O') == 9
def playerChoice(board, player):
        pos = int(input(f'which position do you want {player} ma dude?'))
        if spaceCheck(board,pos) == True:
            #print(Choice)
            place_marker(board,player, pos)
            #return pos
        else: print("that position is not avilable bruh")
def replay():
    play = input("wanna play again Yes/No?")
    return play.lower() == 'yes'
    #return play == 'Yes'
print('Welcome to Tic Tac Toe')
#test_board = [''] * 10
game_on = True
while True:
    test_board = [''] * 10
    display_board(test_board)
    player1 = ''
    player2 = ''
    player1 = player_input(player1)
    player2 = player_input(player2)
    #player_input()
    capture = choose_first(player1, player2, test_board)
    while game_on == True and win_check(test_board, player1) == False and win_check(test_board, player2) == False:
        display_board(test_board)
        if (capture == player1):
            capture = player2
            playerChoice(test_board, capture)
        elif (capture == player2):
            capture = player1
            playerChoice(test_board, capture)
        if fullBoardCheck(test_board) == True:
            print("Game over")
    print('Winner Winner Chicken dinner')
    if replay() == True:
        continue
    else:
        sys.exit()
