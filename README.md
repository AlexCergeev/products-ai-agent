# products-ai-agent

AI-агент, разработанный для снижения количества ошибок при внедрении ИТ-продуктов.

## Описание

Проект направлен на создание AI-агента, который помогает уменьшить количество ошибок в процессе внедрения ИТ-продуктов.

## Где запускать?

Проект может быть запущен:

1. В Colab (самый простой подход, можно запустить одной кнопкой).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexCergeev/products-ai-agent/blob/main/example_agent_colab.ipynb)

2. На личном устройстве.

3. В контуре банка (необходимо установить python 3.11, и получить токен SberOSC)

## Инструкция для запуска на ПК (Ноутбуке)

Чтобы склонировать репозиторий, выполните следующую команду:

```bash
git clone https://github.com/AlexCergeev/products-ai-agent.git
```

### Установка зависимостей

Рекомендуется использовать `pyenv` для управления версиями Python и виртуальным окружением. Для этого выполните следующие шаги:

1. Убедитесь, что у вас установлен `pyenv`. Если нет, установите его согласно [официальной документации](https://github.com/pyenv/pyenv#installation).

2. Установите необходимую версию Python (3.11.4):

    ```bash
    pyenv install 3.11.4
    ```

3. Создайте и активируйте виртуальное окружение:

    ```bash
    pyenv virtualenv 3.11.4 products-ai-agent-env
    pyenv activate products-ai-agent-env
    ```

4. Установите зависимости:

    ```bash
    pip install -r requirements.txt
    ```

## Добавление credentials

Для работы необходимо добавить API-ключи в файл `_config.py`. Создайте этот файл в корневой директории проекта и добавьте в него следующую строку:

```python
# GIGACHAT_API_KEY 
credentials = "your_api_key_here"

confluence_base_url = "confluence_base_url"
jira_base_url = "jira_base_url"

confluence_login = "login"
confluence_password = "password"

jira_login = "login"
jira_password = "password"
```

## Запуск проекта

После установки зависимостей и добавления API-ключа вы можете запустить проект. Например:

```bash
python main.py
```

Агент запросит ссылки на Техническое задание и код проекта (также можно передать требования и код в файлах — их необходимо добавить в корневую директорию и указать их названия).

После этого агент начнет анализ и отправит результаты работы.


Убедитесь, что вы находитесь в активированном виртуальном окружении перед запуском проекта.

