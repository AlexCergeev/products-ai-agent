import os
import logging
import time
import requests
import re
import html

from rag import Rag
from langchain_gigachat.chat_models import GigaChat
from requests.auth import HTTPBasicAuth
from langchain.tools import tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from jira import JIRA 
from bs4 import BeautifulSoup as soup
from urllib.parse import urlparse

credentials = os.environ.get("GIGACHAT_API_KEY")
login = os.environ.get("confluence_login")
password= os.environ.get("confluence_password")
login_jira = os.environ.get("jira_login")
token_jira= os.environ.get("jira_password")

import warnings

warnings.filterwarnings("ignore")

# === Настройка логирования ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ai_agent_calls.log", encoding='utf-8'),
        # logging.StreamHandler()
    ]
)

# === Инициализация модели GigaChat 
gigachat_model = GigaChat(
    credentials=credentials,
    verify_ssl_certs=False,
    timeout=360,
    temperature=0.2,
    top_p=0.5,
    model="GigaChat-Max"
    #max_tokens=10000000
)


# Класс агента
class Agent:
    def __init__(self, role_description, model, max_retries=10, name=None, memory=None):
        """
        Инициализация агента.

        role_description: Описание роли агента.
        model: Модель для выполнения задач (например, GigaChat).
        max_retries: Максимальное количество повторов в случае ошибки.
        name: Имя агента.
        memory: Объект памяти для хранения промежуточных результатов.
        """
        self.role_description = role_description
        self.model = model
        self.max_retries = max_retries
        self.name = name or "Agent"
        self.memory = memory

    def run(self, input_text, memory_key_read=None, memory_key_write=None):
        """
        Выполнение запроса к модели с учетом контекста из памяти.

        input_text: Текст запроса для модели.
        memory_key_read: Ключ памяти, откуда брать контекст.
        memory_key_write: Ключ памяти, куда записывать результат.
        return: Ответ от модели или None в случае ошибки.
        """
        # Получаем содержимое памяти, если указан ключ
        memory_content = ""
        if self.memory and memory_key_read:
            mem = self.memory.read(memory_key_read)
            if mem:
                memory_content = f"\nКонтекст из памяти [{memory_key_read}]:\n{mem}\n"
        
        # Формируем промпт с описанием роли и контекстом
        prompt = f"{self.role_description}\n{memory_content}\n{input_text}"
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.model.invoke(prompt)
                # Извлекаем ответ
                if hasattr(response, "content"):
                    result = response.content.strip()
                else:
                    result = response.strip()
                # Записываем результат в память, если указан ключ для записи
                if self.memory and memory_key_write:
                    self.memory.append(memory_key_write, f"{self.name}:\n{result}")
                return result
            except Exception as e:
                logging.error(f"Ошибка при вызове модели для агента '{self.name}': {e}")
                retries += 1
                time.sleep(20)
                if retries >= self.max_retries:
                    print(f"Ошибка: {str(e)}")
                    return None


# === Определение класса памяти агентов ===
class Memory:
    def __init__(self):
        self.data = {}

    def read(self, key):
        return self.data.get(key, "")

    def append(self, key, value):
        if key in self.data:
            self.data[key] += f"\n{value}"
        else:
            self.data[key] = value

    def clear(self):
        self.data = {}


class Main_Workflow:
    def __init__(self, project_requirements='', project_code='', gigachat_model=gigachat_model):
        """
        Класс работы агентов.

        project_requirements: бизнес требование.
        project_code: код пользователя.
        gigachat_model: модель для агентов.
        """
        self.project_requirements = project_requirements
        self.project_code = project_code
        self.gigachat_model = gigachat_model
        
        
    def extract_id(self, url):
        """
        Извлекает page_id из ссылки на Confluence

        url: ссылка на Confluence.
        return: page_id или None, если page_id не найден
        """
        match = re.search(r'/pages/(\d+)', url)
        return match.group(1) if match else None
    
     
    @staticmethod 
    @tool
    def create_jira_task(link, summary, description):
        """
        Создает задачу в Jira по ссылке link c заголовком summary и описанием description и возвращает её ключ.

        summary: заголовок задачи str
        description: описание задачи str
        return: ключ задачи Jira
        """
        match = re.search(r'/projects/([A-Z0-9]+)', link)
        project_key = match.group(1)
        server_name = urlparse(link).netloc
        jira_options = {'server': f'https://{server_name}'}
        jira = JIRA(options=jira_options, basic_auth=(login_jira, token_jira))
        
        new_issue = jira.create_issue(
                    project=project_key,
                    summary=summary,
                    description=description,
                    issuetype={"name": 'Task'})
        return f"Задача создана {new_issue.key}"
    
    @staticmethod
    @tool
    def create_confluence_comment(link, comment):
        """
        Создает комментарий (comment) в Confluence по ссылке (link).

        Args:
        link: ссылка на confluence str
        comment: комментарий для публикации на confluence str
        """
        # URL для создания комментария
        url = "https://kpaqkpaq.atlassian.net/wiki/rest/api/content"
        # Заголовки
        headers = {
            "Content-Type": "application/json"
        }
        
        match = re.search(r'/pages/(\d+)', link)
        PAGE_ID = match.group(1)
        # Тело запроса
        data = {
            "type": "comment",
            "container": {
                "id": PAGE_ID,
                "type": "page"
            },
            "body": {
                "storage": {
                    "value": html.escape(f"{comment}"),
                    "representation": "storage"
                }
            }
        }
        auth = HTTPBasicAuth(login, password) 
        # Отправка запроса
        response = requests.post(url, auth=auth, headers=headers, json=data)

        return "Комментарий оставлен!"

    def data_read(self, answer_user_material):
        """
        Считывает бизнес требование и код по ссылке Confluenсe или из файла

        answer_user_material: сущность, выделенная агентом, показывающая откуда пользователь хочет загрузить данные
        return: скаченное бизнес требование и код
        """
        # Считывание файла с Confluence
        flag = True
        if 'ссылк' in answer_user_material:
            while flag:
                link = input('Введите ссылку на требования:') 
                PAGE_ID = self.extract_id(link)
                base_url = f"https://kpaqkpaq.atlassian.net/wiki/rest/api/content/{PAGE_ID}?expand=body.storage"
                auth = HTTPBasicAuth(login, password) 
                if PAGE_ID is not None:
                    response = requests.get(base_url, auth=auth)
                    if response.status_code == 200:
                        data = response.json()
                        project_requirements = soup(data['body']['storage']['value'], 'html.parser').get_text(separator="\n", strip=True)
                        flag = False
                    else:
                        print("Ошибка:", response.status_code, response.text)
                else:
                    print('В предоставленной ссылке нет pageId.')
            flag = True
            while flag:
                link = input('Введите ссылку на код:') 
                PAGE_ID = self.extract_id(link)
                base_url = f"https://kpaqkpaq.atlassian.net/wiki/rest/api/content/{PAGE_ID}?expand=body.storage"
                if PAGE_ID is not None:
                    response = requests.get(base_url, auth=auth)
                    if response.status_code == 200:
                        data = response.json()
                        project_code = soup(data['body']['storage']['value'], 'html.parser').get_text(separator="\n", strip=True)
                        flag = False
                    else:
                        print("Ошибка:", response.status_code, response.text)
                else:
                    print('В предоставленной ссылке нет pageId.')
        else:
        # Считывание файла с компьютера
            flag = True
            while flag:
                name = input('Введите название файла с требованиями')
                if not os.path.exists(name):
                    print(f"Файл с требованиями не найден: {name}")
                else:
                    with open(name, 'r', encoding='utf-8') as file:
                        project_requirements = file.read()
                    flag = False
            flag = True
            while flag:
                name = input('Введите название файла с кодом')
                if not os.path.exists(name):
                    print(f"Файл с кодом не найден: {name}")
                else:
                    with open(name, 'r', encoding='utf-8') as file:
                        project_code = file.read()
                    flag = False
        return project_requirements, project_code
            
    def work(self):
        """
        Запуск работы агентов

        return: сохраняет файл с кратким и полным отчетом
        """
        # Создаем общую память для агентов
        shared_memory = Memory()
        rag = Rag()
        # Очистка общей памяти и загрузка исходных данных
        shared_memory.clear()
        results = {}

        # 0. Считывание и проверка входных данных
        # Приветствует и спрашивает, откуда пользователь хочет загрузить данные
        boss_agent = Agent(
            role_description=(
                """Ты руководитель всех AI агентов в данном проекте. 
                Поприветствуй пользователя, расскажи, что ты AI agent, который проверяет бизнес требование, код а также соответсвие кода бизнес требованию. Спроси у пользователя, хочет ли он вставить ссылку на confluence с кодом и бизнес требованием или же хочет загрузить файлы. В ответе укажи только вопрос про файл или ссылку.
                """
            ),
            model=self.gigachat_model,
            memory=shared_memory,
            name="Босс требований"
        )
        
        # Выделяет сущность, откуда пользователь хочет загрузить данные
        wish_checker = Agent(
            role_description=(
                """Ты отлично выделяешь то, что написал пользователь. 
                Задача: Если в ответе пользователя есть что-то похожее на ссылку, то верни в ответе только одно слово - 'ссылку'. Если есть что-то похожее на 'файл', то верни в ответе только одно слово - 'файл'. Если нет ни ссылки ни файла, верни в ответе - "ничего нет"
                """
            ),
            model=GigaChat(
                credentials=credentials,
                verify_ssl_certs=False,
                timeout=360,
                temperature=0.1,
                top_p=0.1,
                model="GigaChat-Max"
            ),
            memory=shared_memory,
            name="Проверка сообщения"
        )
        
         # Выделяет сущность, хочет ли пользователь загрузить данные
        anwer_tool_checker = Agent(
            role_description=(
                """Ты отлично выделяешь хочет ли пользователь что-то использовать или нет. 
                Задача: Если в ответе пользователя есть что-то похожее на желание использовать готовый инструмен, то верни в ответе только одно слово - 'да'. Если желания использовать нет, то верни в ответе - "нет". 
                """
            ),
            model=GigaChat(
                credentials=credentials,
                verify_ssl_certs=False,
                timeout=360,
                temperature=0.1,
                top_p=0.1,
                model="GigaChat-Max"
            ),
            memory=shared_memory,
            name="Проверка желания"
        )

        # Проверяет, что пользователь передал именно Бизнес требование
        req_checker = Agent(
            role_description=(
                """Изучи следующий контекст шаг за шагом:
                1. Сначала внимательно прочитай предоставленный текст
                2. Проверь, является ли предоставленный текст бизнес требованием, а не кодом или просто случайным текстом. Бизнес требование имеет примерно такую структуру: формулирует, что должен делать разработчик, какие результаты ожидать и какие ограничения учитывать.
                3. Если предоставленный текст является бизнес требованием, то в итоговом ответе напиши - 'корректный ввод требований', если нет, напиши - 'некорректный ввод требований'"""
            ),
            model=GigaChat(
                credentials=credentials,
                verify_ssl_certs=False,
                timeout=360,
                temperature=0.1,
                top_p=0.1,
                model="GigaChat-Max"
            ),
            memory=shared_memory,
            name="Андерайтер требований"
        )

        # Проверяет, что пользователь передал именно код
        code_checker = Agent(
            role_description=(
                """Изучи следующий контекст шаг за шагом:
                1. Сначала внимательно прочитай предоставленный текст
                2. Проверь, является ли предоставленный текст кодом на Python, Java, SQL, C++ или Go, а не бизнес требованием или просто случайным текстом.
                3. Если предоставленный текст является кодом, то в итоговом ответе напиши - 'корректный ввод кода', если нет, напиши - 'некорректный ввод кода'"""
            ),
            model=GigaChat(
                credentials=credentials,
                verify_ssl_certs=False,
                timeout=360,
                temperature=0.1,
                top_p=0.1,
                model="GigaChat-Max"
            ),
            memory=shared_memory,
            name="Андерайтер кода"
        )

        # 1. Анализатор требований
        req_analyzer = Agent(
            role_description=(
                "Задача: проанализировать текстовые требования на наличие логических ошибок, двусмысленных формулировок и противоречий. "
                "Выяви нечеткие определения, неопределённые числовые диапазоны, противоречивые условия, а также предложи рекомендации по их исправлению. "
                "Вывод должен содержать список обнаруженных проблем и рекомендации для корректировки требований."
            ),
            model=gigachat_model,
            memory=shared_memory,
            name="Анализатор требований"
        )

        # 2. Анализатор соответствия (сопоставление требований и кода)
        alignment_checker = Agent(
            role_description=(
                "Задача: Сопоставить текстовые требования и исходный код, выявить несоответствия между задокументированным функционалом и реализованными возможностями. "
                "Важно: Если в требованиях описан функционал A, но в коде он отсутствует, это является несоответствием и должно быть указано. "
                "Если в коде присутствует функционал, который не указан в требованиях, это не является ошибкой, так как требования могут быть неполными. "
                "Обрати внимание на отсутствие задокументированных функций, несоответствие числовых диапазонов, а также нарушения архитектурных требований. "
                "Выведи отчет, в котором указаны: требования, которые не реализованы в коде;"
                "а также даны рекомендации по исправлению обнаруженных несоответствий."
            ),
            model=gigachat_model,
            memory=shared_memory,
            name="Анализатор соответствия"
        )

        # 3. Анализатор кодов TODO code filler
        coder = Agent(
            role_description=(
                "Ты умный помощник-программист, который должен генерировать надежный код на Python строго по заданным требованиям. При работе необходимо соблюдать следующие принципы и правила:\n"
                "1. Анализ требований: Перед написанием кода внимательно проанализируй все документированные требования. Убедись, что полностью понял задачу, бизнес-логику и ожидаемый функционал.\n"
                "2. Полнота функционала: Реализуй весь указанный функционал без упущений. Ничего не пропускай – каждая деталь требований должна быть отражена в решении.\n"
                "3. Соответствие логике и ограничениям: Строго соблюдай бизнес-логику, математические формулы и все ограничения, указанные в требованиях. Решение должно точно соответствовать описанным правилам работы.\n",
                "4. В ответе верни только код."
            ),
            model=gigachat_model,
            memory=shared_memory,
            name="Код реализации LLM"
        )

        two_code_analyzer = Agent(
            role_description=(
                "Твоя задача – проанализировать два кода: один предоставлен LLM в качестве справочного бейзлайна, другой – код пользователя. " 
                "Проведи тщательное сравнение с акцентом на математическую корректность реализации: убедись, что диапазоны, знаки в неравенствах и логические условия в коде пользователя полностью совпадают с кодом LLM. " 
                "Отличия в архитектуре, стиле или организации кода не являются ошибками и могут быть проигнорированы. " 
                "В итоговом ответе необходимо сформировать список расхождений, обнаруженных в коде пользователя по сравнению с кодом LLM. " 
                "Код LLM предоставлен исключительно для справки – его комментировать не нужно. Если математическая логика в коде пользователя идентична, выведи сообщение об отсутствии расхождений."
            ),
            model=gigachat_model,
            memory=shared_memory,
            name="Анализатор математической логики"
        )

        # 4. Генератор отчёта
        report_generator = Agent(
            role_description=(
                "Задача: на основе проведенного анализа требований, кода и их соответствия сформировать итоговый отчет, "
                "где перечислены все обнаруженные ошибки, несоответствия и рекомендации по их исправлению. "
                "Включи дополнительную информацию и подробности для каждого найденного пункта. "
                "Вывод должен быть структурированным и понятным для разработчиков и аналитиков."
            ),
            model=self.gigachat_model,
            memory=shared_memory,
            name="Генератор отчёта"
        )

        # 5. Оценщик качества требований и кода
        quality_evaluator = Agent(
            role_description=(
                "Задача: на основе анализа проекта, оценить два аспекта:\n"
                "1. Качество текстовых требований.\n"
                "2. Соответствие исходного кода заявленным требованиям.\n\n"
                "Для каждого аспекта выставь оценку в диапазоне от 0 до 100, где:\n"
                "100% - правильное решение;\n"
                "60% - правильное решение, но есть кейсы, когда решение может быть ошибочным;\n"
                "0% - есть серьезные ошибки, критически влияющие на работу.\n\n"
                "Дай очень короткий комментарий для каждой оценки.\n\n"
                "Выведи ответ строго в следующем формате:\n"
                "Оценка требований: <оценка> - <комментарий>\n"
                "Оценка кода: <оценка> - <комментарий>\n\n"
                "Примеры:\n\n"
                "Пример 1:\n"
                "Требования: \"Функция должна находить максимальное число в списке.\"\n"
                "Код: \"def find_max(lst): return max(lst)\"\n"
                "Вывод: \"Оценка требований: 100% - Требования сформулированы корректно.\\nОценка кода: 100% - Код реализует требуемую функциональность корректно.\"\n\n"
                "Пример 2:\n"
                "Требования: \"Функция должна делить число A на B и возвращать результат.\"\n"
                "Код: \"def divide(a, b): return a / b\"\n"
                "Вывод: \"Оценка требований: 100% - Требования корректны.\\nОценка кода: 60% - Нет проверки деления на ноль, что может привести к ошибке.\"\n\n"
                "Пример 3:\n"
                "Требования: \"Функция должна возвращать сумму чисел в списке.\"\n"
                "Код: \"def sum_numbers(lst): result = 1; for num in lst: result *= num; return result\"\n"
                "Вывод: \"Оценка требований: 100% - Требования сформулированы корректно.\\nОценка кода: 0% - Код реализует умножение вместо сложения, что не соответствует требованию.\""
            ),
            model=self.gigachat_model,
            memory=shared_memory,
            name="Оценщик качества"
        )

        # 6. Суммаризатор
        summarizer_agent = Agent(
            role_description=(
                "Задача: на основе полного отчета, сформированного предыдущими агентами, выделить только самые серьезные ошибки и недочеты. "
                "Сформируй суммаризованный отчет, который должен содержать три части, строго в следующей структуре:\n\n"
                "Ответ по требованиям:\n"
                "*Отчет по требованиям*\n\n"
                "Отчет соответствия требований и кода:\n"
                "*Отчет по коду*\n\n"
                "Оценка:\n"
                "*оценка1*\n"
                "*оценка2*\n\n"
                "В раздел \"Отчет по требованиям\" включи самые критичные проблемы из анализа требований, "
                "в раздел \"Отчет по коду\" – наиболее существенные несоответствия между требованиями и кодом, "
                "а в разделе \"Оценка\" приведи итоговые оценки качества."
            ),
            model=self.gigachat_model,
            memory=shared_memory,
            name="Суммаризатор"
        )
         
        # 7. Агент, который отправляет данные в Jira и Confluence
        agent_jira_confluence = initialize_agent(
            tools=[self.create_jira_task, self.create_confluence_comment],
            llm=gigachat_model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False)
        
        # 8. Работа агентов
        # Приветствует пользователя
        answer_boss_agent = boss_agent.run(
                input_text="",
                memory_key_read=None,
                memory_key_write="Приветсвие пользователя")
            
        print(answer_boss_agent)
        results["Приветствие пользователя"] = answer_boss_agent
        time.sleep(2)

        # Проверяет, что пользователь ввел и сохраняет данные
        answer_user = input()
        answer_user_material = wish_checker.run(input_text=f"n\Ответ от пользователя {answer_user}")
        flag = True
        while flag:
            if 'ничего' not in answer_user_material:
                self.project_requirements, self.project_code = self.data_read(answer_user_material)
                shared_memory.append("Требования пользователя", self.project_requirements)
                shared_memory.append("Код пользователя", self.project_code)
                flag = False
            else:
                answer_user = input('Вы ввели что-то не то, отправьте свой ответ еще раз:')
                answer_user_material = wish_checker.run(input_text=f"n\Ответ от пользователя {answer_user}")
                
        # Проверка кода и ТБ
        flag = True
        while flag:
            # 0. Проверка входных данных
            req_checker_ = req_checker.run(
                input_text="Проверь, является ли предоставленный текст бизнес требованием, а не кодом или просто случайным текстом.",
                memory_key_read="Требования пользователя",
                memory_key_write="Проверка введеных требований"
            )
            results["Проверка введеных требований"] = req_checker_
            time.sleep(5)
        
            code_checker_ = code_checker.run(
                input_text="Проверь, является ли предоставленный текст кодом на Python, Java, SQL, C++ и Go, а не бизнес требованием или просто случайным текстом",
                memory_key_read="Код пользователя",
                memory_key_write="Проверка введеного кода"
            )
            results["Проверка введеного кода"] = code_checker_
            time.sleep(5)
            print(req_checker_.lower())
            print(code_checker_.lower())
            if ('некорректный' in req_checker_.lower()) or ('некорректный' in code_checker_.lower()):
                print('Предоставленные материалы некорректны. Укажите их еще раз.')
                self.project_requirements, self.project_code = self.data_read(answer_user_material)
                shared_memory.append("Требования пользователя", self.project_requirements)
                shared_memory.append("Код пользователя", self.project_code)
            else:
                flag = False
                
        # Запрос в RAG
        data_rag = rag.get_data(self.project_requirements)
        shared_memory.append("Требования пользователя RAG", f"{self.project_requirements}\n{data_rag}")
        
        # Анализ требований
        req_analysis = req_analyzer.run(
            input_text="Проанализируй представленные требования на предмет логических ошибок, двусмысленностей и противоречий.",
            memory_key_read="Требования пользователя RAG",
            memory_key_write="Анализ требований"
        )
        results["Анализ требований"] = req_analysis
        time.sleep(5)
            
        # Сопоставление требований и кода
        # Объединяем исходные требования и код для сравнения
        combined_input = f"Требования:\n{self.project_requirements}\n\nКод:\n{self.project_code}"
        shared_memory.append("Реализация проекта", combined_input)
        alignment_analysis = alignment_checker.run(
            input_text="Сопоставь представленные требования и код, выяви несоответствия (отсутствующие функции, неверные диапазоны, архитектурные нарушения) и дай рекомендации.",
            memory_key_read="Реализация проекта",
            memory_key_write="Анализ соответствия"
        )
        results["Анализ соответствия"] = alignment_analysis
        time.sleep(5)
        
        # Анализатор кодов
        llm_code = coder.run(
            input_text="",
            memory_key_read="Требования пользователя",
            memory_key_write="Код LLM"
        )
        results["Код LLM"] = llm_code

        combined_code = f"Код пользователя:\n{self.project_code}\n\nКод LLM:\n{llm_code}"
        shared_memory.append("Коды", combined_code)
        two_code_analysis = two_code_analyzer.run(
            input_text="Сравни код пользователя и LLM-код по математической корректности и выведи список расхождений или сообщение об их отсутствии, игнорируя стиль и архитектуру.",
            memory_key_read="Коды",
            memory_key_write="Анализ кодов"
        )
        results["Анализ кодов"] = two_code_analysis
        
        # Генерация итогового отчёта
        combined_analysis = f"""Результаты анализа требований:\n{req_analysis}\n
        Результаты сопоставления:\n{alignment_analysis}\n
        Результаты математической корректности:\n{two_code_analysis}\n"""
        
        shared_memory.append("Информация по проекту", combined_analysis)
        time.sleep(5)
        
        # Генерация подробного отчёта
        detail_flag = "Режим: подробный отчет. Включи все подробности по каждому обнаруженному пункту."
        final_report = report_generator.run(
            input_text=detail_flag,
            memory_key_read="Информация по проекту",
            memory_key_write="Отчет"
        )
        time.sleep(5)
        
        # Оценка качества требований и кода
        shared_memory.append("Оценка данных", final_report)
        quality_evaluation = quality_evaluator.run(
            input_text="Оцени соответствие требований и кода, выстави оценку по указанной шкале и дай короткий комментарий.",
            memory_key_read="Оценка данных",
            memory_key_write="Оценка качества"
        )
        
        results["Оценка качества"] = quality_evaluation
        time.sleep(5)
        
        # Добавление оценки качества в конец финального отчёта
        final_report += f"\n\nОценка качества требований и кода:\n{quality_evaluation}"
        results["Итоговый отчет"] = final_report
        time.sleep(5)
        
        # 6. Суммаризация – выделение самых серьезных недочетов и ошибок
        shared_memory.append("Полный отчет", final_report)
        summarized_report = summarizer_agent.run(
            input_text="Сформируй суммаризованный отчет по заданной структуре.",
            memory_key_read="Полный отчет",
            memory_key_write="Суммаризованный отчет"
        )
        results["Суммаризованный отчет"] = summarized_report
        # Спрашиваем пользователя, что он хочет сделать
        answer_conf = input('Хотите ли Вы загрузить данные на конфлюенс?')
        answer_agent = anwer_tool_checker.run(input_text=f"n\Ответ от пользователя {answer_conf}")
        if answer_agent.lower() == 'да':
            while True:
                try:
                    promt = input("Напишите свой запрос агенту. Не забудьте указать ссылку на Confluence:")
                    promt = promt + f' c комментарием: \n\n {results["Суммаризованный отчет"]}'
                    agent_jira_confluence.run(promt)
                    break  # Если ошибок нет, выходим из цикла
                except Exception as e:
                    print(f"Ошибка: {e.__class__.__name__} - {e}. Попробуйте снова.")
        else:
            # Сохраняем краткий отчет в файл            
            with open("Суммаризованный_отчет.txt", "w", encoding='utf-8') as output:
                output.write(results["Суммаризованный отчет"])

            # Сохраняем полный отчет в файл      
            with open("Итоговый_отчет.txt", "w", encoding='utf-8') as output:
                output.write(results["Итоговый отчет"])

        # Спрашиваем пользователя,что он хочет сделать
        answer_jira = input('Хотите ли Вы создать задачу в Jira на доработку?')
        answer_agent = anwer_tool_checker.run(input_text=f"n\Ответ от пользователя {answer_jira}")
        if answer_agent.lower() == 'да':
            while True:
                try:
                    promt = input("Напишите свой запрос агенту. Не забудьте указать ссылку на Jira, заголовок и описание задачи:")
                    agent_jira_confluence.run(promt)
                    break  # Если ошибок нет, выходим из цикла
                except Exception as e:
                    print(f"Ошибка: {e.__class__.__name__} - {e}. Попробуйте снова.")
        else:
            # Сохраняем краткий отчет в файл            
            with open("Суммаризованный_отчет.txt", "w", encoding='utf-8') as output:
                output.write(results["Суммаризованный отчет"])

            # Сохраняем полный отчет в файл      
            with open("Итоговый_отчет.txt", "w", encoding='utf-8') as output:
                output.write(results["Итоговый отчет"])
        
        # Сохраняем краткий отчет в файл            
        with open("Суммаризованный_отчет.txt", "w", encoding='utf-8') as output:
            output.write(results["Суммаризованный отчет"])

        # Сохраняем полный отчет в файл      
        with open("Итоговый_отчет.txt", "w", encoding='utf-8') as output:
            output.write(results["Итоговый отчет"])
                
        print('Краткий и полный отчет сохранены в файл!')
        return