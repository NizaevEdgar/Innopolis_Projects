import pytest
from Books_Collector import BooksCollector
    
class TestBooksCollector:

    
    def test_add_new_book_add_two_books(self):
        # создаем экземпляр (объект) класса BooksCollector
        collector = BooksCollector()

        # добавляем три книги
        collector.add_new_book('Властелин колец')
        collector.add_new_book('Гарри Поттер')
        collector.add_new_book('Матрица')

        # проверяем, что добавилось именно три

        assert len(collector.get_books_genre()) == 3

    
    #Проверяем возможность добавить книгу
    def test_add_new_book(self):
        
        collector = BooksCollector()
        
        collector.add_new_book('Новая книга')
        
        assert collector.get_book_genre('Новая книга') == ''


    #Проверяем возможность установить жанр
    def test_set_book_genre(self):
        
        collector = BooksCollector()
        
        collector.add_new_book('Властелин колец')
        collector.add_new_book('Гарри Поттер')
        collector.add_new_book('Матрица')
        collector.set_book_genre('Властелин колец', 'Фантастика')
        collector.set_book_genre('Гарри Поттер', 'Фэнтези')
        collector.set_book_genre('Матрица', 'Научная фантастика')
        
        assert collector.get_book_genre('Властелин колец') == 'Фантастика'
        assert collector.get_book_genre('Гарри Поттер') == 'Фэнтези'
        assert collector.get_book_genre('Матрица') == 'Научная фантастика'

    #Проверяем добавление нового жанра
    def test_get_books_with_specific_genre(self):
        
        collector = BooksCollector()
        
        collector.add_new_book('Властелин колец')
        collector.add_new_book('Гарри Поттер')
        collector.add_new_book('Матрица')
        collector.set_book_genre('Властелин колец', 'Фантастика')
        collector.set_book_genre('Гарри Поттер', 'Фэнтези')
        collector.set_book_genre('Матрица', 'Научная фантастика')
        
        fantasy_books = collector.get_books_with_specific_genre('Фантастика')
        print(f'Книги в жанре Фантастики: {fantasy_books}')
        
        assert fantasy_books == ['Властелин колец']


    #Есть жанр книги но нет имени (name = None)
    @pytest.mark.parametrize('genre', ['Фантастика', 'Научная фантастика'])
    def test_get_book_genre_with_missing_name(self, genre):
        
        collector = BooksCollector()
        
        collector.add_new_book('Властелин колец')
        book_name = 'Властелин колец'
        collector.set_book_genre('Властелин колец', genre)
        book_genre = collector.get_book_genre(book_name)
        assert book_genre == genre

    #Проверяем, что список книг для детей пуст
    @pytest.mark.parametrize('books_genre, expected_length', [
        ({
             'Оно': 'Ужасы',
             'Матрица': 'Научная фантастика',
             'Властелин колец': 'Фантастика'
         }, 2),
    ])
    def test_get_books_for_children(self, books_genre, expected_length):
        
        collector = BooksCollector()
        
        collector.books_genre = books_genre
        children_books = collector.get_books_for_children()
        print('Книги для детей:', children_books)
        
        assert len(children_books) == expected_length

    #Проверяем, что книга добавляется в Избранное
    @pytest.mark.parametrize("name", [
        ('Властелин колец'),
        ('Гарри Поттер'),
        ('Матрица')
    ])
    
    def test_add_book_in_favorites(self, name):
        
        collector = BooksCollector()
        
        collector.add_new_book(name)
        collector.add_book_in_favorites(name)
        
        assert name in collector.favorites
        assert collector.get_list_of_favorites_books() == [name]

    #Проверяем, что книга удаляется из Избранного
    @pytest.mark.parametrize('name', ['Гарри Поттер'])
    def test_delete_book_from_favorites(self, name):
        
        collector = BooksCollector()
        
        collector.favorites = ['Гарри Поттер']
        collector.delete_book_from_favorites('Гарри Поттер')
        
        assert name not in collector.favorites

    #Проверяем получение списка Избранных книг
    def get_list_of_favorites_books(self):
        return self.favorites