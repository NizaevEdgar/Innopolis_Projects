class CreditCard:
    
    """
    Класс описывает 
    1. Номер кредитной карты
    2. Имя держателя карты
    3. Дату действия карты
    4. CVV код карты
    5. Баланс на карте (по умолчанию 100)
    
    charge - Проводится проверка возможности списания средств с карты
    """
    def __init__(self, card_number, card_holder, expiry_date, cvv, balance=100):
        self.card_number = card_number # Номер карты 
        self.card_holder = card_holder # Имя держателя карты
        self.expiry_date = expiry_date # Дата действия карты 
        self.cvv = cvv # CVV код карты 
        self.balance = balance # Баланс на карте
        
    # Номер карты    
    def getCardNumber(self):
        return self.card_number
    
    # Имя держателя карты
    def getCardHolder(self):
        return self.card_holder
    
    # Дата действия карты    
    def getExpiryDate(self):
        return self.expiry_date
    
    # CVV код карты    
    def getCvv(self):
        return self.cvv
    
    # Сумма платежа
    def charge(self, amount: float):
        if amount <= 0:
            raise ValueError("Сумма платежа должна быть больше нуля")
        if amount > self.balance:
            raise ValueError("Недостаточно средств на карте")
        self.balance -= amount
        return True