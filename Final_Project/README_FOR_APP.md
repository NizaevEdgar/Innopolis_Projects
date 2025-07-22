# 📘 Traffic Sign Detection With YOLO

Это интерактивное приложение на основе Gradio, которое позволяет выполнять детекцию дорожных знаков на изображениях и видео с использованием модели YOLO.

---

<div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">

<div style="flex: 1; min-width: 300px;">
<h3>🚀 Стек технологий</h3>
<ul>
  <li>YOLO (Ultralytics) — модель детекции и классификации объектов</li>
  <li>PyTorch — фреймворк глубокого обучения</li>
  <li>OpenCV — обработка изображений и видео</li>
  <li>Gradio — веб-интерфейс для ML-приложений</li>
  <li>Pillow (PIL) — работа с изображениями</li>
  <li>FFmpeg — обработка аудио и видео</li>
</ul>
</div>

<div style="flex: 1; min-width: 300px;">
<h3>📂 Структура приложения</h3>
<ul>
  <li>Загрузка изображений и видео</li>
  <li>Запуск детекции с настраиваемыми параметрами (порог, толщина, шрифт)</li>
  <li>Сохранение аннотированных файлов</li>
  <li>Предпросмотр результата</li>
  <li>Генерация текстовой легенды сбоку</li>
</ul>
</div>

</div>

---

<div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap; margin-top: 20px;">

<div style="flex: 1; min-width: 300px;">
<h3>🧭 Как пользоваться</h3>
<ol>
  <li>Откройте вкладку <b>Image</b> или <b>Video</b></li>
  <li>Загрузите файл (изображение или видео)</li>
  <li>Настройте параметры:
    <ul>
      <li><b>Confidence Threshold</b> - Порог уверенности</li>
      <li><b>Box Thickness</b> - Толщина рамки</li>
      <li><b>Font Size</b> - Размер шрифта</li>
    </ul>
  </li>
  <li>Нажмите <b>Run</b> для предпросмотра</li>
  <li>Нажмите <b>Save</b> для сохранения результата</li>
  <li>Нажмите <b>Clear</b> для очистки полей</li>
</ol>
</div>

<div style="flex: 1; min-width: 300px;">
<h3>🖼️ Примеры</h3>
<ul>
  <li><code>Example Images</code> — пример изображения дорожного знака</li>
  <li><code>Example Videos</code> — пример видео с реальными условиями</li>
</ul>
</div>

</div>

---
