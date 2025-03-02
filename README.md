# products-ai-agent

AI-агент, разработанный для снижения количества ошибок при внедрении ИТ-продуктов.

## Описание

Проект направлен на создание AI-агента, который помогает уменьшить количество ошибок в процессе внедрения ИТ-продуктов.

## Клонирование репозитория

Чтобы склонировать репозиторий, выполните следующую команду:

```bash
git clone https://github.com/AlexCergeev/products-ai-agent.git
```

## Установка зависимостей

Рекомендуется использовать виртуальное окружение для изоляции зависимостей проекта. Для этого выполните следующие шаги:

1. Перейдите в директорию проекта:

    ```bash
    cd products-ai-agent
    ```

2. Создайте виртуальное окружение с Python 3.11.4:

    ```bash
    python3.11 -m venv env
    ```

3. Активируйте виртуальное окружение:

    - На Windows:

        ```bash
        .\env\Scripts\activate
        ```

    - На macOS и Linux:

        ```bash
        source env/bin/activate
        ```

4. Установите необходимые библиотеки:

    ```bash
    pip install -r requirements.txt
    ```

## Добавление credentials

Для работы с GigaChat необходимо добавить API-ключ в файл `_config.py`. Создайте этот файл в корневой директории проекта и добавьте в него следующую строку:

```python
# GIGACHAT_API_KEY 
credentials = "your_api_key_here"

confluence_base_url = "https://kpaqkpaq.atlassian.net/wiki" # "confluence_base_url"
jira_base_url = "jira_base_url"
# bitbucket_base_url = "bitbucket_base_url"


confluence_login = "login"
confluence_password = "password"

jira_login = "login"
jira_password = "password"

```

Замените `your_api_key_here` на ваш актуальный API-ключ.

## Запуск проекта

После установки зависимостей и добавления API-ключа вы можете запустить проект. Например:

```bash
python main.py
```

Убедитесь, что вы находитесь в активированном виртуальном окружении перед запуском проекта.

