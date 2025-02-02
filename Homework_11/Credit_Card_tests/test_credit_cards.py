import unittest
from credit_cards import CreditCard

class TestCreditCard(unittest.TestCase):
    """Unit тесты для класса CreditCard"""

    def setUp(self):
        """Инициализируем данные перед каждым тестом"""
        self.card = CreditCard("0000 0000 0000 0000", "Ivan Ivanov", "01/30", "123", 500)
        
    def test_getCardNumber(self):
        """Проверяем метод getCardNumber"""
        self.assertEqual(self.card.getCardNumber(), "0000 0000 0000 0000")

    def test_getCardHolder(self):
        """Проверяем метод getCardHolder"""
        self.assertEqual(self.card.getCardHolder(), "Ivan Ivanov")

    def test_getExpiryDate(self):
        """Проверяем метод getExpiryDate"""
        self.assertEqual(self.card.getExpiryDate(), "01/30")

    def test_getCvv(self):
        """Проверяем метод getCvv"""
        self.assertEqual(self.card.getCvv(), "123")
        
    def test_charge_positive(self):
        """Успешное списание"""
        self.assertTrue(self.card.charge(200))
        self.assertEqual(self.card.balance, 300)

    def test_charge_negative(self):
        """Ошибка при списании, недостаточно средств (большая сумма)"""
        with self.assertRaises(ValueError) as context:
            self.card.charge(1500.00)
        self.assertEqual(str(context.exception), "Недостаточно средств на карте")

    def test_charge_zero(self):
        """Ошибка при списании, недостаточно средств (нулевая сумма)"""
        with self.assertRaises(ValueError) as context:
            self.card.charge(0.00)
        self.assertEqual(str(context.exception), "Сумма платежа должна быть больше нуля")

if __name__ == "__main__":
    unittest.main()