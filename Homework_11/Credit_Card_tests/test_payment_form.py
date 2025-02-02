import unittest
from unittest.mock import MagicMock
from payment_form import PaymentForm


class TestPaymentForm(unittest.TestCase):
    """Unit тесты для класса PaymentForm"""

    def setUp(self):
        """Создадим mock объект для CreditCard"""
        self.mock_card = MagicMock()
        self.mock_card.charge.return_value = True
        self.payment_form = PaymentForm(self.mock_card)

    def test_positive_payment(self):
        """Успешная оплата"""
        amount = 500.0
        result = self.payment_form.pay(amount)

        self.mock_card.charge.assert_called_once_with(amount)
        self.assertTrue(result)

    def test_negative_payment(self):
        """Ошибка при списании, недостаточно средств (большая сумма)"""
        self.mock_card.charge.side_effect = ValueError("Недостаточно средств на карте")

        with self.assertRaises(ValueError) as context:
            self.payment_form.pay(1500.0)

        self.assertEqual(str(context.exception), "Недостаточно средств на карте")
        self.mock_card.charge.assert_called_once_with(1500.0)

    def test_zero_amount(self):
        """Ошибка при списании, недостаточно средств (нулевая сумма)"""
        self.mock_card.charge.side_effect = ValueError("Сумма платежа должна быть больше нуля")

        with self.assertRaises(ValueError) as context:
            self.payment_form.pay(0)

        self.assertEqual(str(context.exception), "Сумма платежа должна быть больше нуля")
        self.mock_card.charge.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()