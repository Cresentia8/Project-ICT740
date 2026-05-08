import pygame
import chess
import sys
import os
import numpy as np
import tensorflow as tf

# 1. WINDOWS DLL & PATH CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if sys.platform == 'win32':
    os.add_dll_directory(PROJECT_ROOT)

# 2. PYGAME SETTINGS
WIDTH, HEIGHT = 768, 768
SQUARE_SIZE = WIDTH // 8
WHITE_SQUARE = (240, 217, 181)
BLACK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)
IMAGE_DIR = "OnPC/pieces-basic-png" # Ensure this path is correct

PIECE_IMAGES = {}

# 3. EDGE TPU ENGINE CLASS
class CoralEngine:
    def __init__(self):
        model_path = os.path.join(PROJECT_ROOT, 'OnBorad', 'model', 'ready_for_compiler_edgetpu.tflite')
        delegate = tf.lite.experimental.load_delegate('edgetpu.dll')
        self.interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def encode_board(self, board):
        """Standard 1536 encoding for the TPU model"""
        encoded = np.zeros(1536, dtype=np.int8)
        piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
                     chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                index = piece_map[piece.piece_type]
                if piece.color == chess.BLACK: index += 6
                pos = (square * 12) + index
                if pos < 1536: encoded[pos] = 1
        return encoded.reshape(1, 1536)

    def get_best_move(self, board):
        input_data = self.encode_board(board)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Strategy: Pick the highest probability LEGAL move
        best_indices = np.argsort(prediction)[::-1]
        for idx in best_indices:
            from_sq = int(idx // 64)
            to_sq = int(idx % 64)
            move = chess.Move(from_sq, to_sq)
            # Handle promotions if the model suggests moving a pawn to the end
            if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN:
                if chess.square_rank(to_sq) in [0, 7]:
                    move.promotion = chess.QUEEN
            
            if move in board.legal_moves:
                return move
        return None

# 4. DRAWING FUNCTIONS
def load_images():
    piece_files = {
        'p': 'black-pawn.png', 'r': 'black-rook.png', 'n': 'black-knight.png',
        'b': 'black-bishop.png', 'q': 'black-queen.png', 'k': 'black-king.png',
        'P': 'white-pawn.png', 'R': 'white-rook.png', 'N': 'white-knight.png',
        'B': 'white-bishop.png', 'Q': 'white-queen.png', 'K': 'white-king.png'
    }
    for symbol, filename in piece_files.items():
        path = os.path.join(IMAGE_DIR, filename)
        image = pygame.image.load(path)
        PIECE_IMAGES[symbol] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(screen, selected_square):
    for row in range(8):
        for col in range(8):
            color = WHITE_SQUARE if (row + col) % 2 == 0 else BLACK_SQUARE
            if selected_square is not None:
                if col == chess.square_file(selected_square) and (7 - row) == chess.square_rank(selected_square):
                    color = HIGHLIGHT_COLOR
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol()
            x = chess.square_file(square) * SQUARE_SIZE
            y = (7 - chess.square_rank(square)) * SQUARE_SIZE
            screen.blit(PIECE_IMAGES[symbol], (x, y))

# 5. MAIN LOOP
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Coral Edge TPU Chess AI")
    
    load_images()
    engine = CoralEngine()
    board = chess.Board()
    selected_square = None
    running = True

    while running:
        # Check if it's AI's turn (Black)
        if not board.is_game_over() and board.turn == chess.BLACK:
            print("AI is thinking...")
            ai_move = engine.get_best_move(board)
            if ai_move:
                board.push(ai_move)
            else:
                print("AI is stuck!")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                col, row = pos[0] // SQUARE_SIZE, pos[1] // SQUARE_SIZE
                clicked_square = chess.square(col, 7 - row)

                if selected_square is None:
                    piece = board.piece_at(clicked_square)
                    if piece and piece.color == board.turn:
                        selected_square = clicked_square
                else:
                    move = chess.Move(selected_square, clicked_square)
                    # Auto-promote to Queen for simplicity
                    if board.piece_at(selected_square).piece_type == chess.PAWN:
                        if chess.square_rank(clicked_square) in [0, 7]:
                            move.promotion = chess.QUEEN
                    
                    if move in board.legal_moves:
                        board.push(move)
                    selected_square = None

        draw_board(screen, selected_square)
        draw_pieces(screen, board)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()