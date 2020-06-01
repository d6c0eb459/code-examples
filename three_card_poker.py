"""
Three Card Poker is a casino game based on poker.

In Three Card Poker, the player and the dealer each draw a hand of three cards from a
single shuffled deck. The two hands are compared using poker rules: the best hand is a
straight flush (three cards in order ex. Jack, Queen, King all of the same suit ex.
all Hearts), followed by three of a kind (ex. three Jacks), straight (ex. two, three
four), flush (ex. all Hearts), any pair, and lastly high card. Ties are broken by the
next highest card (ex. three Queens beat three Jacks, or a pair of Jacks with a King
beats a pair of Jacks with a Queen). The suit is not used to break ties.

Given two hands of three cards, we are interested in determining:
    1. Which hand is better, or if it is a tie.
    2. A human readable printout for result of the game including their hand, who won,
        and the class of hand they won with (ex. flush, pair).

The following code example is done in functional programming style, complete with unit
tests.

Requires Python 3.6 (for f string formatting), should pass pylint and black
autoformatting.

Note that there are similarities with this game and the better known Fizz buzz problem
https://en.wikipedia.org/wiki/Fizz_buzz
"""
import random
import unittest
import unittest.mock


def hand_create(indexes):
    """
    Convert a list of cards represented by index to a list of (suit, rank) tuples.

    Rank is represented as an integer from 0 to 12 representing, in order, the ranks 2,
    3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, and Ace.

    Suit is represented as an integer from 0 to 3 repesenting, in order, Spades, Hearts,
    Diamonds, Clubs.

    Arguments:
        indexes list(int)
            A list of indexes, where indexes 0 to 12 inclusive represent Spades, 13 to 25
            represent Hearts, etc.

    Returns: list(tuple(int, int))
        The given list of cards converted to list of (suit, rank) tuples.
    """
    output = []
    for index in indexes:
        index = index % 52
        rank = index % 13
        suit = index // 13
        output.append((suit, rank))
    return output


def hand_describe(hand):
    """
    Convert a hand to a human readable string.

    Rank is represented as an integer from 0 to 12 representing, in order, the ranks 2,
    3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, and Ace.

    Suit is represented as an integer from 0 to 3 repesenting, in order, Spades, Hearts,
    Diamonds, Clubs.

    Arguments:
        hand list(tuple(int, int)):
            A hand of cards, each in (suit, rank) format.

    Returns: str
        A text description of the hand.
    """
    texts = []
    for (suit, rank) in hand:
        if rank == 9:
            rank = "J"
        elif rank == 10:
            rank = "Q"
        elif rank == 11:
            rank = "K"
        elif rank == 12:
            rank = "A"
        else:
            rank = str(rank + 2)

        suit = ["♠", "♥", "♦", "♣"][suit]

        texts.append(f"{rank}{suit}")

    return " ".join(texts)


def rating_create(hand):
    """
    Assign a rating code to a hand of Three Card Poker.

    The rating code is a list where the first element represents the class of hand
    (straight flush, triple, etc.) and the second and third elements represent tie
    breaker elements for that class. For example, if the first element encodes that the
    hand has a pair, the second element encodes what rank that pair is.

    The rating code is designed such that two codes can be easily compared using standard
    string comparison and no additional logic.

    Arguments:
        hand list(tuple(int, int)):
            A hand of three cards, each in (suit, rank) format.

    Returns: list(int)
        A rating code.
    """
    # pylint: disable=too-many-return-statements
    ranks = sorted((card[1] for card in hand), reverse=True)
    suits = list(card[0] for card in hand)

    is_straight = ranks[0] == ranks[1] + 1 and ranks[1] == ranks[2] + 1
    is_flush = suits[0] == suits[1] and suits[1] == suits[2]

    if is_straight and is_flush:
        return [5] + ranks

    is_triple = ranks[0] == ranks[1] and ranks[1] == ranks[2]
    if is_triple:
        return [4] + ranks

    if is_straight:
        return [3] + ranks

    if is_flush:
        return [2] + ranks

    is_front_pair = ranks[0] == ranks[1]
    if is_front_pair:
        # Ties are broken based on who has the higher pair, then who has the higher third
        # card, which is the last card in this case.
        return [1, ranks[0], ranks[2]]

    is_back_pair = ranks[1] == ranks[2]
    if is_back_pair:
        # Ties are broken based on who has the higher pair, then who has the higher third
        # card, which is the first card in this case.
        return [1, ranks[1], ranks[0]]

    return [0] + ranks


def rating_describe(rating):
    """
    Human readable rating for a rating.

    Arguments:
        rating list(int):
            A rating code for the hand.

    Returns: str
        A human readable description of the rating.
    """
    cls = rating[0]  # Class of the hand
    if cls == 5:
        return "straight flush"

    if cls == 4:
        return "triple"

    if cls == 3:
        return "straight"

    if cls == 2:
        return "flush"

    if cls == 1:
        return "pair"

    if cls == 0:
        return "high card"

    raise ValueError


def play_three_card_poker():
    """
    Play Thee Card Poker!

    The player and dealer each draw cards, which are compared using poker rules.

    Returns: list(str)
        A list of human readable descriptions of how the round of Three Card Poker went,
        one per line.
    """
    cards = random.sample(range(52), 6)
    player_cards = cards[:3]
    dealer_cards = cards[3:]

    player_hand = hand_create(player_cards)
    player_rating = rating_create(player_hand)

    dealer_hand = hand_create(dealer_cards)
    dealer_rating = rating_create(dealer_hand)

    # Dealer needs at least a Queen to play
    dealer_qualifies = dealer_rating >= rating_create(hand_create([49, 0, 1]))

    player_description = rating_describe(player_rating)
    dealer_description = rating_describe(dealer_rating)

    output = []
    output.append(f"Player's cards: {hand_describe(player_hand)}")
    output.append(f"Dealer's cards: {hand_describe(dealer_hand)}")

    if not dealer_qualifies:
        output.append("Play bets push as dealer does not have a Queen or better.")
    if player_rating > dealer_rating:
        output.append(
            (
                f"Player's {player_description} wins "
                f"against dealer's {dealer_description}!"
            )
        )
    elif player_rating < dealer_rating:
        output.append(
            (
                f"Player's {player_description} loses "
                f"against dealer's {dealer_description}."
            )
        )
    else:
        output.append(
            (
                f"Player's {player_description} ties "
                f"with dealer's {dealer_description}."
            )
        )

    return output


class TestHand(unittest.TestCase):
    """
    Tests for hand functions.
    """

    def test_rating_create(self):
        """
        Test the hand rating function.
        """
        # Straight flush
        actual = rating_create(hand_create([1, 2, 3]))
        expected = [5, 3, 2, 1]
        self.assertEqual(actual, expected)

        # Three of a kind
        actual = rating_create(hand_create([0, 13, 26]))
        expected = [4, 0, 0, 0]
        self.assertEqual(actual, expected)

        # Straight
        actual = rating_create(hand_create([0, 14, 28]))
        expected = [3, 2, 1, 0]
        self.assertEqual(actual, expected)

        # Flush
        actual = rating_create(hand_create([0, 2, 4]))
        expected = [2, 4, 2, 0]
        self.assertEqual(actual, expected)

        # Front pair
        actual = rating_create(hand_create([0, 13, 1]))
        expected = [1, 0, 1]
        self.assertEqual(actual, expected)

        # Back pair
        actual = rating_create(hand_create([1, 13, 0]))
        expected = [1, 0, 1]
        self.assertEqual(actual, expected)

        # High card
        actual = rating_create(hand_create([0, 30, 51]))
        expected = [0, 12, 4, 0]
        self.assertEqual(actual, expected)

        # No wrap-around (twos are worse than Aces)
        actual = rating_create(hand_create([11, 12, 13]))
        expected = [0, 12, 11, 0]
        self.assertEqual(actual, expected)

    def test_rating_describe(self):
        """
        Test the hand description function.
        """
        actual = rating_describe([5])
        expected = "straight flush"
        self.assertEqual(actual, expected)

        actual = rating_describe([4])
        expected = "triple"
        self.assertEqual(actual, expected)

        actual = rating_describe([3])
        expected = "straight"
        self.assertEqual(actual, expected)

        actual = rating_describe([2])
        expected = "flush"
        self.assertEqual(actual, expected)

        actual = rating_describe([1])
        expected = "pair"
        self.assertEqual(actual, expected)

        actual = rating_describe([0])
        expected = "high card"
        self.assertEqual(actual, expected)

    def test_hand_describe(self):
        """
        Test the hand describe
        """
        # Test ranks
        actual = hand_describe(hand_create(range(13)))
        expected = "2♠ 3♠ 4♠ 5♠ 6♠ 7♠ 8♠ 9♠ 10♠ J♠ Q♠ K♠ A♠"
        self.assertEqual(actual, expected)

        # Test suits
        actual = hand_describe(hand_create([0, 13, 26, 39]))
        expected = "2♠ 2♥ 2♦ 2♣"
        self.assertEqual(actual, expected)

    def test_play_three_card_poker(self):
        """
        Test the play function.
        """
        # Lose
        with unittest.mock.patch.object(random, "sample") as mock_random:
            mock_random.return_value = [0, 1, 2, 3, 4, 5]
            actual = play_three_card_poker()

        expected = [
            "Player's cards: 2♠ 3♠ 4♠",
            "Dealer's cards: 5♠ 6♠ 7♠",
            "Player's straight flush loses against dealer's straight flush.",
        ]
        self.assertEqual(actual, expected)

        # Win
        with unittest.mock.patch.object(random, "sample") as mock_random:
            mock_random.return_value = [12, 25, 0, 3, 11, 24]
            actual = play_three_card_poker()

        expected = [
            "Player's cards: A♠ A♥ 2♠",
            "Dealer's cards: 5♠ K♠ K♥",
            "Player's pair wins against dealer's pair!",
        ]
        self.assertEqual(actual, expected)

        # Tie
        with unittest.mock.patch.object(random, "sample") as mock_random:
            mock_random.return_value = [11, 2, 16, 24, 2, 3]
            actual = play_three_card_poker()

        expected = [
            "Player's cards: K♠ 4♠ 5♥",
            "Dealer's cards: K♥ 4♠ 5♠",
            "Player's high card ties with dealer's high card.",
        ]
        self.assertEqual(actual, expected)

        # Dealer doesn't qualify
        with unittest.mock.patch.object(random, "sample") as mock_random:
            mock_random.return_value = [11, 2, 16, 0, 7, 14]
            actual = play_three_card_poker()

        expected = [
            "Player's cards: K♠ 4♠ 5♥",
            "Dealer's cards: 2♠ 9♠ 3♥",
            "Play bets push as dealer does not have a Queen or better.",
            "Player's high card wins against dealer's high card!",
        ]
        self.assertEqual(actual, expected)

        # Dealer barely qualifies
        with unittest.mock.patch.object(random, "sample") as mock_random:
            mock_random.return_value = [11, 2, 16, 10, 0, 14]
            actual = play_three_card_poker()

        expected = [
            "Player's cards: K♠ 4♠ 5♥",
            "Dealer's cards: Q♠ 2♠ 3♥",
            "Player's high card wins against dealer's high card!",
        ]
        self.assertEqual(actual, expected)

    def test_hand_compare(self):
        """
        Test the hand comparison function.
        """
        # High straight flush wins
        alpha = rating_create(hand_create([10, 11, 12]))
        beta = rating_create(hand_create([1, 2, 3]))
        self.assertGreater(alpha, beta)

        # Straight flush ties even if different suit
        alpha = rating_create(hand_create([0, 1, 2]))
        beta = rating_create(hand_create([13, 14, 15]))
        self.assertEqual(alpha, beta)

        # High front pair wins
        alpha = rating_create(hand_create([12, 25, 0]))
        beta = rating_create(hand_create([11, 24, 0]))
        self.assertGreater(alpha, beta)

        # High back pair wins
        alpha = rating_create(hand_create([1, 14, 2]))
        beta = rating_create(hand_create([0, 13, 2]))
        self.assertGreater(alpha, beta)

        # High card wins tiebreak
        alpha = rating_create(hand_create([0, 12, 11]))
        beta = rating_create(hand_create([0, 12, 10]))
        self.assertGreater(alpha, beta)

        # High card wins
        alpha = rating_create(hand_create([0, 5, 11]))
        beta = rating_create(hand_create([0, 4, 10]))
        self.assertGreater(alpha, beta)


if __name__ == "__main__":
    for line in play_three_card_poker():
        print(line)

    unittest.main()
