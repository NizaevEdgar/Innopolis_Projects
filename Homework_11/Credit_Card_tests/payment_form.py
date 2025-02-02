from credit_cards import CreditCard

class PaymentForm:
    def __init__(self, credit_card):
        self.credit_card = credit_card
    
    def pay(self, amount: float):
        return self.credit_card.charge(amount)