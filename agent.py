import os
import logging
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import pwinput
from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage

from _config import credentials

# === Настройка логирования ===
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ai_agent_calls.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# === Настройка визуализации ===
console = Console()

# === Настройка модели GigaChat (или вашей модели) ===
# Запрашиваем API-ключ для доступа к модели

gigachat_model = GigaChat(
    credentials=credentials,
    verify_ssl_certs=False,
    timeout=120,
    temperature=0.3,
    top_p=0.5,
    model="GigaChat-Max",
    # model="GigaChat",
    # max_tokens=500,
)

# === Определение класса памяти ===
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

# Создаем общую память для агентов
shared_memory = Memory()

# === Определение базового агента ===
class Agent:
    def __init__(self, role_description, model, memory=None, max_retries=3, name=None):
        """
        Инициализация агента.
        :param role_description: Описание роли агента.
        :param model: Модель для выполнения задач (например, GigaChat).
        :param memory: Объект памяти для хранения и чтения данных.
        :param max_retries: Максимальное количество повторов в случае ошибки.
        :param name: Имя агента.
        """
        self.role_description = role_description
        self.model = model
        self.memory = memory
        self.max_retries = max_retries
        self.name = name or "Agent"

    def run(self, input_text, memory_key_read='shared_context', memory_key_write='shared_context'):
        """
        Выполнение запроса к модели с учётом информации из памяти.
        :param input_text: Текст запроса для модели.
        :param memory_key_read: Ключ памяти для чтения.
        :param memory_key_write: Ключ памяти для чтения.
        :return: Ответ от модели или None в случае ошибки.
        """
        # Получаем содержимое памяти, если оно есть
        memory_content = ""
        if self.memory:
            memory_content = self.memory.read(memory_key_read)
            if memory_content:
                memory_content = f"\nКонтекст из памяти [{memory_key_read}]:\n{memory_content}\n"
        
        # Формируем промпт с описанием роли и контекстом
        prompt = f"{self.role_description}\n{memory_content}\n{input_text}"
        retries = 0
        while retries < self.max_retries:
            try:
                console.print(Panel(f"[bold yellow]Промпт агента {self.name}:[/bold yellow]\n{prompt}", border_style="bright_blue"))
                logging.info(f"Попытка #{retries + 1} отправки запроса для агента '{self.name}'")
                response = self.model.invoke(prompt)
                if hasattr(response, "content"):
                    result = response.content.strip()
                else:
                    result = response.strip()
                logging.info(f"Ответ от модели для агента '{self.name}': {result}")
                console.print(Panel(f"[bold green]Ответ {self.name}:[/bold green]\n{result}", border_style="bright_green"))
                # Записываем результат в память
                if self.memory:
                    self.memory.append(memory_key_write, f"{self.name}:\n{result}")
                return result
            except Exception as e:
                logging.error(f"Ошибка при вызове модели для агента '{self.name}': {e}")
                retries += 1
                if retries >= self.max_retries:
                    console.print(Panel(f"[bold red]Ошибка:{str(e)}[/bold red]", border_style="red"))
                    return None

# === Определение агентов по ролям ===

# 0. Экстракторы 

# code_finder = Agent(
#     role_description=(
# '''Задача: определить, содержит ли входной текст фрагменты программного кода.

# Инструкции:
# 1. Прочитай входной текст полностью.
# 2. Определи, присутствуют ли в тексте признаки программного кода. Признаками кода являются:
#    - Структурированные фрагменты с отступами, ключевыми словами, операторами, скобками и другими элементами, характерными для языков программирования.
#    - Фрагменты, предназначенные для выполнения программой (например, объявления переменных, функции, циклы, условные операторы).
# 3. Исключи математические формулы:
#    - Если в тексте присутствуют только математические формулы (например, оформленные в LaTeX: между знаками `$...$`, `$$...$$`, `$begin:math:text$...$end:math:text$` или `$begin:math:display$...$end:math:display$`), их не следует считать программным кодом.
# 4. Если обнаружены признаки программного кода, не являющиеся исключительно математическими формулами, выведи единственный ответ: "Да".
# 5. Если признаки программного кода отсутствуют или присутствуют только математические формулы, выведи единственный ответ: "Нет".
# 6. Игнорируй весь остальной текст и не добавляй никаких дополнительных пояснений.'''
#     ),
#     model=gigachat_model,
#     memory=shared_memory,
#     name="Наличие кода"
# )

# code_finder = Agent(
#     role_description=('''Задача: очистить входной текст, содержащий обычный текст и программный код, оставив только программный код в неизменном виде.
# Инструкции:
# 1. Прочитай весь входной текст.
# 2. Выдели все участки, которые являются программным кодом. Удали весь остальной текст.
# 3. В выделенных участках кода удали только комментарии. Комментарии могут быть:
#    - Однострочными (например, начинающимися с `//` или `#`);
#    - Многострочными (например, заключёнными в `/* ... */`).
# 4. **Очень важно:** не изменяй исходный код. Сохрани все отступы, форматирование, структуру и порядок строк без каких-либо модификаций. Единственное допустимое изменение — удаление комментариев.
# 5. Выведи итоговый результат, содержащий исключительно чистый код без обычного текста и комментариев.'''),
#     model=gigachat_model,
#     memory=shared_memory,
#     name="Код"
# )

# text_finder = Agent(
#     role_description=('''Тебе на вход подается большой файл, содержащий обычный текст и программный код. 
# Твоя задача — удалить из файла весь программный код, оставив исключительно текст. 
# При этом текст не изменяй. Выведи итоговый результат.'''),
#     model=gigachat_model,
#     memory=shared_memory,
#     name="Текст"
# )

# 1. Планировщик
planner = Agent(
    role_description=(
        "Вы опытный планировщик проекта. Ваша задача — разработать структурированный план для анализа артефактов IT-проекта. "
        "Определите последовательность действий для проверки соответствия требований, user story, кода и тестовых сценариев, "
        "с учётом снижения количества ошибок и оптимизации процессов."
    ),
    model=gigachat_model,
    memory=shared_memory,
    name="Планировщик"
)

# # 2. Аналитик требований
# req_analyst = Agent(
#     role_description=(
#         "Вы аналитик требований. Ваша задача — внимательно проанализировать представленные требования и user story. "
#         "Выявите ключевые цели, критические моменты и потенциальные источники ошибок, которые могут привести к несоответствиям."
#     ),
#     model=gigachat_model,
#     memory=shared_memory,
#     name="Аналитик требований"
# )

# 3. Программист
programmer = Agent(
    role_description=("""Ты опытный программист. Тебе переданы требования для разработки. Пошагово:
1. Проанализируй каждый пункт требований.
2. Реализуй каждый пункт, соблюдая логику и математические расчёты.
3. Не изменяй алгоритмы и структуру решения.
4. Выведи итоговый результат исключительно в виде кода, без каких-либо комментариев или пояснений."""),
    model=gigachat_model,
    memory=shared_memory,
    name="Код решения робота"
)

# 4. Аналитик соответствия кода и требований
alignment_analyst = Agent(
    role_description=(
        '''Задача: сравнить результаты анализа требований и анализа кода, выявить несоответствия между задекларированными требованиями и реализованным функционалом, а также обозначить потенциальные риски.
Инструкции:
1. Прочитай оба раздела входного текста: анализ требований и анализ кода.
2. Выдели ключевые требования, указанные в разделе требований.
3. Определи реализованный функционал на основе анализа кода.
4. Сравни задекларированные требования с реализованным функционалом:
   - Если обнаружены расхождения или отсутствует реализованный функционал, требуемый по требованиям, зафиксируй это как несоответствие.
   - Если в коде присутствуют функции, не указанные в требованиях, отметь это как отклонение.
5. Оцени потенциальные риски, связанные с выявленными несоответствиями (например, риски для безопасности, производительности или надежности).
6. Сформируй итоговый отчет, в котором:
   - Перечислены все выявленные несоответствия между требованиями и кодом.
   - Описаны потенциальные риски, вытекающие из этих несоответствий.
7. Выведи только итоговый отчет без дополнительных комментариев или пояснений.'''),
    model=gigachat_model,
    memory=shared_memory,
    name="Аналитик соответствия"
)

code_analyst = Agent(
    role_description=(
'''Ты опытный программист. Тебе даны два кода: один от человека, один от робота. Твоя задача:
1. Сравнить математическую корректность обоих кодов.
2. Учесть, что архитектура может отличаться, но логика и математические операции должны совпадать.
3. Если найдены несоответствия или ошибки в коде человека, опиши их по пунктам.
4. Особое внимание удели знакам и логическим условиям, так как люди часто допускают ошибки в этих местах.
5. Выведи результаты анализа, перечислив найденные ошибки и несоответствия.
6. Укажи только на ошибки человека'''),
    model=gigachat_model,
    memory=shared_memory,
    name="Аналитик кода"
)
# 5. Критик
critic = Agent(
    role_description=(
        '''Ты критик системы. Проанализируй все полученные результаты и укажи критические ошибки, риски и недостатки. 
Ответ должен быть кратким и точным.'''
    ),
    model=gigachat_model,
    memory=shared_memory,
    name="Критик"
)

# 6. Итоги
summary = Agent(
    role_description=(
'''Задача: проанализировать входной текст, содержащий требования и/или код, и выявить только ошибки. Комментарии, пояснения и дополнительные примечания игнорировать.
Инструкции:
1. Прочитай и проанализируй входной текст.
2. Выдели все обнаруженные ошибки в коде пользователя:
   - Ошибки могут быть математическими, логическими, синтаксическими, нарушениями требований и прочими недочётами.
3. Игнорируй любой текст, являющийся комментариями или пояснениями (например, строки, начинающиеся с символов комментария или явно помеченные как комментарии).
4. Выведи только список ошибок (каждая ошибка на отдельной строке или в виде нумерованного/маркированного списка).
5. Если ошибок не обнаружено, выведи: "Ошибок не найдено".'''
    ),
    model=gigachat_model,
    memory=shared_memory,
    name="Выводы"
)

# === Основной рабочий процесс ===
def main_workflow():
    """
    Основной рабочий процесс мультиагентной системы для анализа артефактов IT-проекта.
    """
    console.print("[bold cyan]Добро пожаловать в систему анализа артефактов IT-проекта![/bold cyan]", justify="center")
    console.print("[bold green]Пожалуйста, введите описание проекта, включающее требования, user story, код и тестовые сценарии:[/bold green]")
    with open('/Users/kpaq/Documents/code/products-ai-agent/test_er3_in_border_ages_text.txt', 'r') as file:
        project_description = file.read()
        
    with open('/Users/kpaq/Documents/code/products-ai-agent/test_er3_in_border_ages_code2.txt', 'r') as file:
        project_code = file.read()

    # project_description = console.input(">> ")
    results = {}
    shared_memory.clear()
    # shared_memory.append('Контент пользователя', f"Запрос пользователя:\n{project_description}")
    results["Требования пользователя"] = project_description
    shared_memory.append("Требования пользователя", f"\n{project_description}")
    
    results["Код пользователя"] = project_code
    shared_memory.append("Код пользователя", f"\n{project_code}")
    # Инициализируем общую память с исходным описанием проекта

    # # 0. Экстракторы 
    # # code_finder_ = code_finder.run("", memory_key='shared_context')
    # # results["Наличие:"] = code_finder_
    # text_finder_ = text_finder.run("", memory_key_read='Контент пользователя')
    # results["Требования пользователя"] = text_finder_
    # shared_memory.append("Требования пользователя", f"\n{results['Требования пользователя']}")
    
    # code_finder_ = code_finder.run("", memory_key_read='Контент пользователя')
    # results["Код пользователя"] = code_finder_
    # shared_memory.append("Код пользователя", f"\n{results['Код пользователя']}")
    
    
    # 1. Планировщик: разрабатывает общий план анализа
    # plan = planner.run("Разработайте подробный план анализа артефактов для выявления несоответствий и потенциальных ошибок.", memory_key='shared_context')
    # results["Планировщик"] = plan

    # # 2. Аналитик требований: анализирует требования и user story
    # req_analysis = req_analyst.run("Проанализируйте требования и user story, выделите ключевые цели и потенциальные проблемные зоны.", memory_key='shared_context')
    # results["Аналитик требований"] = req_analysis

    # # 3. Программист
    # code_analysis = code_analyst.run("Проанализируйте исходный код проекта на предмет ошибок, нарушений лучших практик и потенциальных уязвимостей.", memory_key='shared_context')
    # results["Аналитик кода"] = code_analysis
    
    
    code_llm = programmer.run("", memory_key_read='Требования пользователя')
    results["Код робота"] = code_llm
    shared_memory.append("Код робота", f"\n{results['Код робота']}")
    

    # 4. Аналитики
    shared_memory.append("Коды", f"\nКод пользователя\n{results['Код пользователя']}\nКод робота:\n{results['Код робота']}")
    code_analysis = code_analyst.run("", memory_key_read='Коды')
    results["Аналитик кода"] = code_analysis
    
    shared_memory.append("Реализация пользователя", f"\Требования\n{results['Требования пользователя']}\nКод:\n{results['Код пользователя']}")
    alignment_analysis = alignment_analyst.run("", memory_key_read='Реализация пользователя')
    results["Аналитик соответствия"] = alignment_analysis
    
    # 5. Критик: проводит итоговую оценку и предлагает рекомендации по улучшению
    # critic_analysis = critic.run("На основе всех предыдущих анализов, оцените общую картину, укажите критические ошибки и предложите способы их устранения.", memory_key='shared_context')
    # results["Критик"] = critic_analysis
    shared_memory.append("Информация по проекту", f"\n{results['Аналитик соответствия']}\n{results['Аналитик кода']}")
    sum_analysis = summary.run("", memory_key_read='Информация по проекту')
    results["Выводы"] = sum_analysis

    # Вывод финального отчёта
    console.print("\n[bold cyan]Финальный отчет по анализу артефактов:[/bold cyan]\n")
    for role, outcome in results.items():
        console.print(Panel(f"{outcome}", title=f"[bold green]{role}[/bold green]", border_style="bright_magenta"))

if __name__ == "__main__":
    main_workflow()