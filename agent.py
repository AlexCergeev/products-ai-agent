import os
import logging
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import pwinput
from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
import time
from _config import credentials

# === Настройка логирования ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ai_agent_calls.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# === Настройка визуализации ===
console = Console()

# === Инициализация модели GigaChat (или другой выбранной модели) ===
gigachat_model = GigaChat(
    credentials=credentials,
    verify_ssl_certs=False,
    timeout=360,
    temperature=0.2,
    top_p=0.5,
    model="GigaChat-Max",
    max_tokens=10000000
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
    def __init__(self, role_description, model, max_retries=3, name=None, memory=None):
        """
        Инициализация агента.
        :param role_description: Описание роли агента.
        :param model: Модель для выполнения задач (например, GigaChat).
        :param max_retries: Максимальное количество повторов в случае ошибки.
        :param name: Имя агента.
        :param memory: Объект памяти для хранения промежуточных результатов.
        """
        self.role_description = role_description
        self.model = model
        self.max_retries = max_retries
        self.name = name or "Agent"
        self.memory = memory

    def run(self, input_text, memory_key_read=None, memory_key_write=None):
        """
        Выполнение запроса к модели с учетом контекста из памяти.
        :param input_text: Текст запроса для модели.
        :param memory_key_read: Ключ памяти, откуда брать контекст.
        :param memory_key_write: Ключ памяти, куда записывать результат.
        :return: Ответ от модели или None в случае ошибки.
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
                console.print(Panel(f"[bold yellow]Промпт агента {self.name}:[/bold yellow]\n{prompt}", border_style="bright_blue"))
                logging.info(f"Попытка #{retries + 1} отправки запроса для агента '{self.name}'")
                response = self.model.invoke(prompt)
                # Извлекаем ответ
                if hasattr(response, "content"):
                    result = response.content.strip()
                else:
                    result = response.strip()
                logging.info(f"Ответ от модели для агента '{self.name}': {result}")
                console.print(Panel(f"[bold green]Ответ {self.name}:[/bold green]\n{result}", border_style="bright_green"))
                # Записываем результат в память, если указан ключ для записи
                if self.memory and memory_key_write:
                    self.memory.append(memory_key_write, f"{self.name}:\n{result}")
                return result
            except Exception as e:
                logging.error(f"Ошибка при вызове модели для агента '{self.name}': {e}")
                retries += 1
                if retries >= self.max_retries:
                    console.print(Panel(f"[bold red]Ошибка: {str(e)}[/bold red]", border_style="red"))
                    return None

# === Определение агентов по ролям ===

# 1. Анализатор требований
req_analyzer = Agent(
    role_description=(
        "Задача: проанализировать текстовые требования на наличие логических ошибок, двусмысленных формулировок и противоречий. "
        "Выяви нечеткие определения, неопределённые числовые диапазоны, противоречивые условия, а также предложи рекомендации по их исправлению. "
        "Вывод должен содержать краткий список обнаруженных проблем и рекомендации для корректировки требований."
    ),
    model=gigachat_model,
    memory=shared_memory,
    name="Анализатор требований"
)

# 2. Анализатор кода
code_analyzer = Agent(
    role_description=(
        "Задача: проанализировать исходный код, написанный на Python, Java, SQL, C++ и Go, на предмет логических и функциональных ошибок. "
        "Проверь соответствие вычислительных диапазонов, корректность логических условий и архитектурных решений. "
        "Если обнаружены ошибки (например, неверное сравнение, отсутствие проверки граничных условий, несоответствие реализованной логики требованиям), "
        "выведи список проблем с рекомендациями по их исправлению. Фокус – только на логических и функциональных ошибках."
    ),
    model=gigachat_model,
    memory=shared_memory,
    name="Анализатор кода"
)

# 3. Анализатор соответствия (сопоставление требований и кода)
alignment_checker = Agent(
    role_description=(
        "Задача: сопоставить текстовые требования и исходный код, выявить несоответствия между задокументированным функционалом и реализованными возможностями. "
        "Обрати внимание на отсутствие функций, несоответствие числовых диапазонов, а также нарушения архитектурных требований. "
        "Выведи отчет, где указаны: какие требования не реализованы, какие функции присутствуют в коде, но не описаны в требованиях, "
        "а также дай рекомендации по исправлению обнаруженных несоответствий."
    ),
    model=gigachat_model,
    memory=shared_memory,
    name="Анализатор соответствия"
)

# 4. Генератор отчёта
report_generator = Agent(
    role_description=(
        "Задача: на основе проведенного анализа требований, кода и их соответствия сформировать итоговый краткий отчет, "
        "где перечислены все обнаруженные ошибки, несоответствия и рекомендации по их исправлению. "
        "Если запрошен подробный отчет (режим детально), включи дополнительную информацию и подробности для каждого найденного пункта. "
        "Вывод должен быть структурированным и понятным для разработчиков и аналитиков."
    ),
    model=gigachat_model,
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
    model=gigachat_model,
    memory=shared_memory,
    name="Оценщик качества"
)

# 6. Суммаризатор
summarizer_agent = Agent(
    role_description=(
        "Задача: на основе полного отчета, сформированного предыдущими агентами, выделить только самые серьезные недочеты и ошибки. "
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
    model=gigachat_model,
    memory=shared_memory,
    name="Суммаризатор"
)


# === Основной рабочий процесс ===
def main_workflow():
    """
    Основной рабочий процесс мультиагентной системы для анализа текстовых требований и исходного кода.
    """
    console.print("[bold cyan]Добро пожаловать в систему анализа соответствия требований и кода![/bold cyan]", justify="center")
    
    # Чтение файла с текстовыми требованиями
    requirements_file_path = '/Users/kpaq/Documents/code/products-ai-agent/tests/БТ2/БТ2.txt'  # Укажите корректный путь к файлу требований
    if not os.path.exists(requirements_file_path):
        console.print(f"[bold red]Файл с требованиями не найден: {requirements_file_path}[/bold red]")
        return
    with open(requirements_file_path, 'r', encoding='utf-8') as file:
        project_requirements = file.read()
        
    # Чтение файла с исходным кодом
    code_file_path = '/Users/kpaq/Documents/code/products-ai-agent/tests/БТ2/Код2.txt'  # Укажите корректный путь к файлу с кодом (или реализуйте обход нескольких файлов)
    if not os.path.exists(code_file_path):
        console.print(f"[bold red]Файл с кодом не найден: {code_file_path}[/bold red]")
        return
    with open(code_file_path, 'r', encoding='utf-8') as file:
        project_code = file.read()
    
    # Очистка общей памяти и загрузка исходных данных
    shared_memory.clear()
    shared_memory.append("Требования пользователя", project_requirements)
    shared_memory.append("Код пользователя", project_code)
    
    results = {}
    
    # 1. Анализ требований
    req_analysis = req_analyzer.run(
        input_text="Проанализируй представленные требования на предмет логических ошибок, двусмысленностей и противоречий.",
        memory_key_read="Требования пользователя",
        memory_key_write="Анализ требований"
    )
    results["Анализ требований"] = req_analysis
    time.sleep(5)
    # 2. Анализ кода
    code_analysis = code_analyzer.run(
        input_text="Проанализируй исходный код на наличие логических и функциональных ошибок, а также несоответствий архитектурным требованиям.",
        memory_key_read="Код пользователя",
        memory_key_write="Анализ кода"
    )
    results["Анализ кода"] = code_analysis
    time.sleep(5)
    # 3. Сопоставление требований и кода
    # Объединяем исходные требования и код для сравнения
    combined_input = f"Требования:\n{project_requirements}\n\nКод:\n{project_code}"
    shared_memory.append("Реализация проекта", combined_input)
    alignment_analysis = alignment_checker.run(
        input_text="Сопоставь представленные требования и код, выяви несоответствия (отсутствующие функции, неверные диапазоны, архитектурные нарушения) и дай рекомендации.",
        memory_key_read="Реализация проекта",
        memory_key_write="Анализ соответствия"
    )
    results["Анализ соответствия"] = alignment_analysis
    time.sleep(5)
    # 4. Генерация итогового отчёта
    combined_analysis = (
        f"Результаты анализа требований:\n{req_analysis}\n\n"
        f"Результаты анализа кода:\n{code_analysis}\n\n"
        f"Результаты сопоставления:\n{alignment_analysis}"
    )
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
    # 5. Оценка качества требований и кода
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
    
    # Вывод финального отчёта в консоль
    console.print("\n[bold cyan]Финальный отчет по анализу требований и кода:[/bold cyan]\n")
    for role, outcome in results.items():
        console.print(Panel(f"{outcome}", title=f"[bold green]{role}[/bold green]", border_style="bright_magenta"))

if __name__ == "__main__":
    main_workflow()