import telebot
from config import API_TOKEN, YANDEX_FOLDER_ID, YANDEX_API_KEY, ADMIN_CHAT_ID, SUPPORT_CHANNEL_ID
import re
import os
import time
import glob
import json
import random
import requests
import datetime
from telebot.types import BotCommand, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton, ForceReply, Message
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from yandex_cloud_ml_sdk import YCloudML
import threading

# Testing mode configuration
# When True, all interface buttons are disabled and users can only ask questions to the AI
TESTING_MODE = False

# Helper functions to toggle testing mode
def enable_testing_mode():
    """Enable testing mode - disables all UI and commands except direct AI interaction"""
    global TESTING_MODE
    TESTING_MODE = True
    print("Testing mode ENABLED - interface and commands disabled")
    try:
        bot.delete_my_commands()
        print("Removed all commands for testing mode")
    except Exception as e:
        print(f"Error removing commands: {str(e)}")
    return True

def disable_testing_mode():
    """Disable testing mode - restores normal UI and commands"""
    global TESTING_MODE
    TESTING_MODE = False
    print("Testing mode DISABLED - interface and commands restored")
    setup_bot_commands()
    return True

bot = telebot.TeleBot(API_TOKEN)

QDRANT_DB_PATH = "/QDRANT/qdrant_data_USER-bge-m3_2000-200"
DOCUMENTS_PATH = "documents"
SUPPORT_DATA_FILE = "support_data.json"
LOGS_FILE = "query_logs.json"
os.makedirs(QDRANT_DB_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

print(QDRANT_DB_PATH)

print("RAG system initialization...")
qdrant_client = QdrantClient(path=QDRANT_DB_PATH)

encoder = SentenceTransformer("deepvk/USER-bge-m3")

yandex_sdk = YCloudML(
    folder_id=YANDEX_FOLDER_ID,
    auth=YANDEX_API_KEY,
)

available_documents = []
user_support_chats = {}
user_states = {}  # Values: "initial", "normal", "support", "selecting_ticket", "documents"

def save_support_data():
    try:
        data_to_save = {}
        for user_id, chat_data in user_support_chats.items():
            data_to_save[str(user_id)] = chat_data
        
        with open(SUPPORT_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"Support data saved to {SUPPORT_DATA_FILE}")
    except Exception as e:
        print(f"Error saving support data: {str(e)}")

def load_support_data():
    global user_support_chats
    try:
        if os.path.exists(SUPPORT_DATA_FILE):
            with open(SUPPORT_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for user_id_str, chat_data in data.items():
                    user_support_chats[int(user_id_str)] = chat_data
                
                print(f"Loaded {len(user_support_chats)} active support tickets from {SUPPORT_DATA_FILE}")
        else:
            print(f"No support data file found at {SUPPORT_DATA_FILE}")
    except Exception as e:
        print(f"Error loading support data: {str(e)}")
        user_support_chats = {}

def delete_ticket_from_json(topic_id):
    try:
        user_id_to_remove = None
        
        for uid, data in user_support_chats.items():
            if data.get("topic_message_id") == topic_id:
                user_id_to_remove = uid
                break
        
        if user_id_to_remove is None:
            print(f"No user found for topic ID {topic_id}")
            return False
            
        if os.path.exists(SUPPORT_DATA_FILE):
            try:
                with open(SUPPORT_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if str(user_id_to_remove) in data:
                    data.pop(str(user_id_to_remove))
                    
                    with open(SUPPORT_DATA_FILE, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    print(f"Ticket {topic_id} removed from {SUPPORT_DATA_FILE}")
                    return True
                else:
                    print(f"User ID {user_id_to_remove} not found in support data file")
                    return False
            except Exception as e:
                print(f"Error updating support data file: {str(e)}")
                return False
        else:
            print(f"Support data file {SUPPORT_DATA_FILE} not found")
            return False
    except Exception as e:
        print(f"Error deleting ticket from JSON: {str(e)}")
        return False

load_support_data()
 
def get_main_menu_keyboard():
    """Returns the main menu keyboard with support button."""
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [
        KeyboardButton("💬 Задать вопрос ИИ"),
        KeyboardButton("📞 Поддержка ЧатБота"),
        KeyboardButton("📚 Документы")
    ]
    keyboard.add(*buttons)
    return keyboard

def get_cancel_keyboard():
    """Returns a keyboard with just a cancel button."""
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    keyboard.add(KeyboardButton("❌ Отмена"))
    return keyboard

def get_support_menu_keyboard(has_active_ticket=False):
    """Returns the support menu keyboard."""
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = []
    
    # Only show "New request" if user doesn't have an active ticket
    if not has_active_ticket:
        buttons.append(KeyboardButton("📝 Создать обращение в поддержку"))
    
    buttons.append(KeyboardButton("👋 Выйти на главное меню"))
    
    if has_active_ticket:
        buttons.insert(0, KeyboardButton("🎫 Мое активное обращение"))
    
    keyboard.add(*buttons)
    return keyboard

def remove_headers_footers(text: str) -> str:
    """Remove headers and footers without affecting main text."""
    patterns = [
        r'Европейское соглашение о международной дорожной перевозке \s*опасных грузов" \(ДОПОГ\/ADR\).*?\n',
        r'\(заключено в г\. Женеве 30\.09\.1957\.\.\. .*?\n',
        r'Документ предоставлен КонсультантПлюс.*?\n',
        r'Дата сохранения: \d{2}\.\d{2}\.\d{4}.*?\n',
        r'КонсультантПлюс.*?надежная правовая поддержка.*?\n',
        r'Страница \d+ из \d+.*?\n',
        r'ГОСТ Р 57478-2017\. Национальный стандарт Российской Федерации\.\s*Грузы опасные\. Классификация.*?\n',
        r'\(утв\. и введен в действие \.\.\. .*?\n',
        r'Copyright\s*©\s*United\s*Nations[^\n]*\d{4}[^\n]*All\s*rights\s*reserved[^\n]*',
        r'©\s*United\s*Nations[^\n]*\d{4}[^\n]*',
        r'Copyright\s*©\s*[^\n]*UN[^\n]*\d{4}[^\n]*',
        r'-\s*\d+\s*-',
        r'—\s*\d+\s*—',
        r'Страница \d+ из \d+.*?\n',
        r'Документ предоставлен КонсультантПлюс.*?\n',
        r'Дата сохранения: \d{2}\.\d{2}\.\d{4}.*?\n'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    text = re.sub(r'\n+', ' ', text)  # Replace hyphens with spaces
    text = re.sub(r'\s+', ' ', text)  # Removing multiple spaces
 
    return text.strip()

def process_pdf_files(
    file_paths: List[str],
    start_pages: Optional[List[Optional[int]]] = None,
    page_offsets: Optional[List[int]] = None,
    chunk_size: int = 2000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Process multiple PDF files with individual page numbering offsets.

    Args:
        file_paths: List of PDF file paths
        start_pages: List of starting pages (None to process from first page)
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        page_offsets: List of page numbering offsets (one per file)

    Returns:
        List of processed documents with correct page numbering
    """
    if start_pages is None:
        start_pages = [1] * len(file_paths)
    elif len(start_pages) != len(file_paths):
        raise ValueError("Количество стартовых страниц должно совпадать с количеством файлов")

    if page_offsets is None:
        page_offsets = [0] * len(file_paths)
    elif len(page_offsets) != len(file_paths):
        raise ValueError("Количество смещений должно совпадать с количеством файлов")

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    for file_path, start_page, offset in zip(file_paths, start_pages, page_offsets):
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        current_start_page = 1 if start_page is None else start_page
        processed_pages = []

        for i, page in enumerate(pages, start=1):
            if i >= current_start_page:
                metadata = page.metadata.copy()

                if 'page_label' in metadata:
                    try:
                        original_label = int(metadata['page_label'])
                        metadata['page_label'] = str(original_label + offset)
                    except ValueError:
                        pass

                if 'page' in metadata:
                    metadata['page'] += offset

                if 'total_pages' in metadata:
                    metadata['total_pages'] += offset

                processed_pages.append(Document(
                    page_content=remove_headers_footers(page.page_content),
                    metadata=metadata
                ))

        chunks = text_splitter.split_documents(processed_pages)
        all_chunks.extend(chunks)

    return all_chunks

def remove_duplicate_chunks(chunks: List[Document]) -> List[Document]:
    """Remove duplicate chunks by text content, keeping the first occurrence.
    Normalizes text before comparison for better deduplication."""
    seen_texts = set()
    unique_chunks = []
    
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    for doc in chunks:
        normalized_content = normalize_text(doc.page_content)
        
        if len(normalized_content) < 20:
            continue
            
        if normalized_content not in seen_texts:
            seen_texts.add(normalized_content)
            unique_chunks.append(doc)
    
    #print(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks, {len(unique_chunks)} remaining")
    return unique_chunks

def remove_duplicate_search_results(search_results: List[Dict]) -> List[Dict]:
    """
    Remove duplicates from Qdrant search results.
    Compares texts by normalizing them for better duplicate detection.
    """
    if not search_results:
        return []
        
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    normalized_texts = []
    for item in search_results:
        content = item['payload'].get('page_content', '')
        normalized_texts.append(normalize_text(content))
    
    # Удаляем дубликаты, сохраняя тот экземпляр с более высоким рейтингом релевантности
    unique_results = []
    seen_texts = set()
    
    # Сортируем результаты по релевантности (score) от большей к меньшей
    sorted_results = sorted(zip(search_results, normalized_texts), key=lambda x: x[0]['score'], reverse=True)
    
    for result, norm_text in sorted_results:
        if len(norm_text) < 20:
            continue
            
        # Проверяем, нет ли уже очень похожего текста в результатах
        if norm_text not in seen_texts:
            seen_texts.add(norm_text)
            unique_results.append(result)
    
    return unique_results

def get_document_config():
    """Returns document configuration for processing."""
    document_files = sorted(glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf")))
    
    # Проверяем наличие всех документов
    required_docs = ["ПДД.pdf", "ГОСТ-Р-57478-2017.pdf", 
                     "Европейское_соглашение_о_международной_дорожной_перевозке_опасных.pdf",
                     "ДОПОГ_том_1.pdf", "ДОПОГ_том_2.pdf"]
    
    # Получаем имена файлов в нижнем регистре для проверки
    doc_basenames = [os.path.basename(f).lower() for f in document_files]
    
    # Проверяем наличие каждого требуемого документа
    start_pages = []
    page_offsets = []
    


    # Базовые настройки смещения страниц для документов
    config_map = {
        "ПДД.pdf": (0, 1),
        "ГОСТ-Р-57478-2017.pdf": (2, 1),
        "Европейское_соглашение_о_международной_дорожной_перевозке_опасных.pdf": (1, 1),
        "ДОПОГ_том_1.pdf": (26, 2),
        "ДОПОГ_том_2.pdf": (16, 3)
    }

   
    # Применяем настройки к найденным документам
    for doc_path in document_files:
        doc_name = os.path.basename(doc_path).lower()
        
        # По умолчанию начинаем с первой страницы и без смещения
        sp, po = None, 0
        
        # Ищем в конфигурации
        for key, (start_page, page_offset) in config_map.items():
            if key in doc_name or any(k in doc_name for k in key.split('-')):
                sp, po = start_page, page_offset
                break
        
        start_pages.append(sp)
        page_offsets.append(po)
    
    return document_files, start_pages, page_offsets

def initialize_rag():
    """Initialize RAG system by creating or loading Qdrant collection."""
    
    try:
        # Проверяем существование коллекции
        collections = qdrant_client.get_collections()
        collection_exists = any(collection.name == "docs" for collection in collections.collections)
        
        if collection_exists:
            print("Existing RAG database from Qdrant loaded...")
            return
        
        print("Creating new RAG database...")
        # Создаем коллекцию
        qdrant_client.recreate_collection(
            collection_name="docs",
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        
        # Получаем все PDF файлы и их конфигурацию
        pdf_files, start_pages, page_offsets = get_document_config()
        
        if not pdf_files:
            print("No PDF files found in the documents directory!")
            return
        
        #print(f"Found {len(pdf_files)} PDF files")
        #print(f"Start pages configuration: {start_pages}")
        print(f"Page offsets configuration: {page_offsets}")
        
        # Обрабатываем PDF файлы с настроенными параметрами
        chunks = process_pdf_files(pdf_files, start_pages, page_offsets)
        
        # Удаляем дублирующиеся чанки
        chunks = remove_duplicate_chunks(chunks)
        
        if not chunks:
            print("No valid chunks found in documents!")
            return
            
        # Загружаем данные в Qdrant
        records = []
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            vector = encoder.encode(chunk.page_content).tolist()
            
            records.append(models.Record(
                id=idx,
                vector=vector,
                payload={
                    "metadata": chunk.metadata,
                    "page_content": chunk.page_content
                }
            ))
            
            # Загружаем батчами по 100 для экономии памяти
            if len(records) >= 100:
                qdrant_client.upload_records(
                    collection_name="docs",
                    records=records
                )
                records = []

        
        # Загружаем оставшиеся записи
        if records:
            qdrant_client.upload_records(
                collection_name="docs",
                records=records
            )
        
        print("RAG system initialized and saved to disk")
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        try:
            qdrant_client.recreate_collection(
                collection_name="docs",
                vectors_config=models.VectorParams(
                    size=encoder.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            print("Created empty fallback collection")
        except:
            print("Critical error: Could not create fallback collection")

initialize_rag()

# Function to verify support channel setup
def verify_support_channel_setup():
    try:
        # First check if the bot can access the channel
        api_url = f"https://api.telegram.org/bot{API_TOKEN}/getChat"
        payload = {
            "chat_id": SUPPORT_CHANNEL_ID
        }
        
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                chat_data = result.get("result", {})
                
                # Check if this is a supergroup
                if not chat_data.get("type") == "supergroup":
                    print("⚠️ Warning: Support channel is not a supergroup")
                    print("Please convert the channel to a supergroup in Telegram settings")
                    return False
                
                # Check if forum is enabled
                is_forum = chat_data.get("is_forum", False)
                if not is_forum:
                    print("⚠️ Warning: Forum feature is not enabled in the support channel")
                    print("Please enable 'Topics' in the channel settings")
                    return False
                
                print("✅ Support channel is properly configured as a forum")
                return True
            else:
                print(f"❌ Error getting chat info: {result.get('description', 'Unknown error')}")
                return False
        else:
            print(f"❌ API request failed with status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error verifying support channel setup: {str(e)}")
        return False

# Function to verify bot permissions in the support channel
def verify_support_channel_permissions():
    try:
        # Check if the bot is an admin in the support channel
        api_url = f"https://api.telegram.org/bot{API_TOKEN}/getChatMember"
        payload = {
            "chat_id": SUPPORT_CHANNEL_ID,
            "user_id": bot.get_me().id
        }
        
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                chat_member = result.get("result", {})
                status = chat_member.get("status", "")
                
                if status in ["creator", "administrator"]:
                    # Check if the bot has can_manage_topics permission
                    can_manage_topics = chat_member.get("can_manage_topics", False)
                    
                    if can_manage_topics:
                        print("✅ Bot has proper permissions in the support channel")
                        return True
                    else:
                        print("⚠️ Warning: Bot is an admin but doesn't have 'can_manage_topics' permission in the support channel")
                        print("Please update bot permissions in the support channel")
                        return False
                else:
                    print("⚠️ Warning: Bot is not an administrator in the support channel")
                    print("Please add the bot as an administrator to the support channel with 'can_manage_topics' permission")
                    return False
            else:
                print(f"❌ Error checking bot permissions: {result.get('description', 'Unknown error')}")
                return False
        else:
            print(f"❌ API request failed with status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error verifying support channel permissions: {str(e)}")
        return False

# Check support channel configuration
support_channel_properly_setup = False
try:
    if verify_support_channel_setup() and verify_support_channel_permissions():
        support_channel_properly_setup = True
        print("✅ Support channel is properly configured and bot has correct permissions")
    else:
        print("⚠️ Support channel is not properly configured. Support features will be limited.")
except Exception as e:
    print(f"Failed to verify support channel setup: {str(e)}")

# Set up bot commands menu
def setup_bot_commands():
    # In testing mode, only set up minimal commands
    if TESTING_MODE:
        print("Testing mode enabled - setting up minimal commands")
        minimal_commands = [
            BotCommand("start", "Запустить бота"),
            BotCommand("documents", "Показать список доступных документов для скачивания")
        ]
        try:
            bot.set_my_commands(minimal_commands)
            print(f"Configured minimal testing mode commands: {', '.join(cmd.command for cmd in minimal_commands)}")
        except Exception as e:
            print(f"Error setting minimal commands: {str(e)}")
        return
    
    print("Setting up bot commands...")
    
    # Main commands for users
    commands = [
        BotCommand("start", "Запустить бота"),
        BotCommand("documents", "Показать список доступных документов для скачивания"),
        BotCommand("support", "Связаться с поддержкой чатбота"),
        BotCommand("cancel", "Выйти из режима с поддержкой чатбота"),
    ]
    bot.set_my_commands(commands)
    print(f"Configured general commands: {', '.join(cmd.command for cmd in commands)}")
    
    # Set commands for support channel (only visible in the channel)
    support_commands = [
        BotCommand("active", "Показать активные запросы в поддержку"),
        BotCommand("close", "Закрыть тикет (используйте в треде тикета)")
    ]
    try:
        bot.set_my_commands(support_commands, scope=telebot.types.BotCommandScopeChat(SUPPORT_CHANNEL_ID))
        print(f"Configured support channel commands: {', '.join(cmd.command for cmd in support_commands)}")
        print(f"Support channel ID: {SUPPORT_CHANNEL_ID}")
    except Exception as e:
        print(f"Error setting support channel commands: {str(e)}")
        
    # Set commands for admin
    admin_commands = [
        BotCommand("start", "Запустить бота"),
        BotCommand("documents", "Показать список доступных документов для скачивания"),
        BotCommand("support", "Связаться с поддержкой чатбота"),
        BotCommand("cancel", "Выйти из режима с поддержкой чатбота"),
    ]
    try:
        bot.set_my_commands(admin_commands, scope=telebot.types.BotCommandScopeChat(ADMIN_CHAT_ID))
        print(f"Configured admin commands: {', '.join(cmd.command for cmd in admin_commands)}")
    except Exception as e:
        print(f"Error setting admin commands: {str(e)}")

# Setup commands menu on startup
setup_bot_commands()

# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    chat_id = message.chat.id
    
    # If testing mode is enabled, show simplified welcome and no buttons
    if TESTING_MODE:
        # Set user state to normal to directly process questions
        user_states[chat_id] = "normal"
        bot.reply_to(message, """\
Привет! Вы участвуете в тестировании бота.

Просто задайте любой вопрос и получите ответ найденный в документах!
""", reply_markup=telebot.types.ReplyKeyboardRemove())
        return
    
    # Regular welcome message for normal mode
    user_states[chat_id] = "initial"
    
    bot.reply_to(message, """\
Привет, я бот, работающий под управлением ИИ.

Задайте мне любой вопрос, основанный на доступных документах, и я постараюсь предоставить информацию основанную на них!
""", reply_markup=get_main_menu_keyboard())


# Function to get user info string
def get_user_info_str(message):
    user = message.from_user
    user_info = f"<b>Пользователь:</b> {user.first_name}"
    if user.last_name:
        user_info += f" {user.last_name}"
    if user.username:
        user_info += f" (@{user.username})"
    user_info += f"\n<b>ID:</b> <code>{message.chat.id}</code>"
    return user_info

# Handle '/support' command
@bot.message_handler(commands=['support'])
def start_support_chat(message):
    chat_id = message.chat.id
    
    # Set user state to support mode
    user_states[chat_id] = "support"
    
    # Check if the support channel is properly setup
    if not support_channel_properly_setup:
        bot.reply_to(message, """\
❌ Система поддержки не настроена или настроена неправильно.

Для настройки системы поддержки:
1. Создайте группу/канал с включенными темами (topics)
2. Добавьте бота как администратора с правом управления темами
3. Обновите ID канала в настройках бота
""", reply_markup=get_main_menu_keyboard())
        return
    
    # Check if the user already has an active ticket
    has_active_ticket = chat_id in user_support_chats and user_support_chats[chat_id].get("in_support_mode", False)
    
    # Show support menu
    bot.send_message(
        chat_id,
        "Вы перешли в режим технической поддержки. Выберите действие:",
        reply_markup=get_support_menu_keyboard(has_active_ticket)
    )

# Handle new support request
def create_new_support_request(message):
    chat_id = message.chat.id
    
    force_reply = ForceReply(selective=True)
    prompt_msg = bot.send_message(
        chat_id,
        "Опишите вашу проблему или задайте вопрос технической поддержке:",
        reply_markup=force_reply
    )
    
    # Store temporary state
    user_support_chats[chat_id] = {
        "in_support_mode": False,  # Not yet in support mode
        "prompt_message_id": prompt_msg.message_id,
        "user_info": get_user_info_str(message),
        "username": message.from_user.username or "",
        "first_name": message.from_user.first_name,
        "last_name": message.from_user.last_name or "",
    }
    
    # Save updated support data
    save_support_data()

# Handle viewing active support ticket
def view_active_ticket(message):
    chat_id = message.chat.id
    
    # Get all active tickets for this user
    active_tickets = []
    
    # Check for tickets in the user_support_chats
    if chat_id in user_support_chats and user_support_chats[chat_id].get("in_support_mode", False):
        # Add the current chat's ticket
        ticket_data = user_support_chats[chat_id]
        topic_id = ticket_data.get("topic_message_id")
        ticket_id = topic_id if topic_id else ticket_data.get("direct_ticket_id", "Unknown")
        active_tickets.append({
            "ticket_id": ticket_id,
            "topic_msg_id": topic_id,
            "time_created": ticket_data.get("time_created", 0)
        })
    
    # Debug info for tickets
    print(f"Active tickets found for user {chat_id}: {len(active_tickets)}")
    for ticket in active_tickets:
        print(f"  Ticket: {ticket}")
    
    # If no active tickets
    if not active_tickets:
        bot.send_message(
            chat_id,
            "У вас нет активных запросов в поддержку. Вы можете создать новый запрос.",
            reply_markup=get_support_menu_keyboard(False)
        )
        return
    
    # If only one ticket, show it directly
    if len(active_tickets) == 1:
        ticket = active_tickets[0]
        keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
        buttons = [
            KeyboardButton("❌ Закрыть тикет"),
            KeyboardButton("👋 Выйти из поддержки")
        ]
        keyboard.add(*buttons)
        
        # Format creation time
        creation_time = "недавно"
        if "time_created" in user_support_chats[chat_id]:
            elapsed = time.time() - user_support_chats[chat_id]["time_created"]
            if elapsed < 60:
                creation_time = f"{int(elapsed)} сек назад"
            elif elapsed < 3600:
                creation_time = f"{int(elapsed / 60)} мин назад"
            elif elapsed < 86400:
                creation_time = f"{int(elapsed / 3600)} ч назад"
            else:
                creation_time = f"{int(elapsed / 86400)} дн назад"
        
        bot.send_message(
            chat_id,
            f"🎫 <b>Ваш активный тикет #{ticket['ticket_id']}</b>\n"
            f"Создан: {creation_time}\n\n"
            f"Вы можете продолжать общение с технической поддержкой. Просто напишите ваше сообщение.",
            parse_mode="HTML",
            reply_markup=keyboard
        )
        
        # Set user state to support mode with this ticket
        user_states[chat_id] = "support"
    else:
        # Multiple tickets - show selection keyboard
        keyboard = InlineKeyboardMarkup(row_width=1)
        
        # Sort tickets by creation time (newest first)
        active_tickets.sort(key=lambda x: x.get("time_created", 0), reverse=True)
        
        for idx, ticket in enumerate(active_tickets, 1):
            ticket_id = ticket["ticket_id"]
            # Format creation time
            creation_time = "недавно"
            if "time_created" in user_support_chats[chat_id]:
                elapsed = time.time() - user_support_chats[chat_id]["time_created"]
                if elapsed < 60:
                    creation_time = f"{int(elapsed)} сек назад"
                elif elapsed < 3600:
                    creation_time = f"{int(elapsed / 60)} мин назад"
                elif elapsed < 86400:
                    creation_time = f"{int(elapsed / 3600)} ч назад"
                else:
                    creation_time = f"{int(elapsed / 86400)} дн назад"
            
            button_text = f"Тикет #{ticket_id} (создан {creation_time})"
            keyboard.add(InlineKeyboardButton(button_text, callback_data=f"ticket_{ticket_id}"))
        
        bot.send_message(
            chat_id,
            "У вас несколько активных тикетов. Выберите, к какому тикету вы хотите перейти:",
            reply_markup=keyboard
        )
        
        # Set state to selecting ticket
        user_states[chat_id] = "selecting_ticket"

# Handle ticket selection via inline keyboard
@bot.callback_query_handler(func=lambda call: call.data.startswith('ticket_'))
def handle_ticket_selection(call):
    chat_id = call.message.chat.id
    ticket_id = call.data.split('_')[1]
    
    # Update the message
    bot.edit_message_text(
        f"Выбран тикет #{ticket_id}",
        chat_id=chat_id,
        message_id=call.message.message_id
    )
    
    # Create keyboard with ticket options - only Close ticket and Exit support
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [
        KeyboardButton("❌ Закрыть тикет"),
        KeyboardButton("👋 Выйти из поддержки")
    ]
    keyboard.add(*buttons)
    
    bot.send_message(
        chat_id,
        f"🎫 <b>Ваш активный тикет #{ticket_id}</b>\n\n"
        f"Вы можете продолжать общение с технической поддержкой. Просто напишите ваше сообщение.",
        parse_mode="HTML",
        reply_markup=keyboard
    )
    
    # Set user state to support mode with this ticket
    user_states[chat_id] = "support"

# Handle '/cancel' command to exit support mode
@bot.message_handler(commands=['cancel'])
def cancel_support_chat(message):
    chat_id = message.chat.id
    
    # Reset state to normal
    user_states[chat_id] = "normal"
    
    if chat_id in user_support_chats and user_support_chats[chat_id].get("in_support_mode", False):
        # Get topic information
        topic_msg_id = user_support_chats[chat_id].get("topic_message_id")
        
        # Notify user but don't close the ticket
        bot.send_message(
            chat_id,
            f"Вы вышли из режима поддержки. Ваш тикет #{topic_msg_id if topic_msg_id else user_support_chats[chat_id].get('direct_ticket_id', 'Unknown')} остается активным, и вы получите уведомление при ответе от поддержки.",
            reply_markup=get_main_menu_keyboard()
        )
    else:
        bot.reply_to(message, "У вас нет активных запросов в поддержку.", reply_markup=get_main_menu_keyboard())

# Handle replies to the support prompt
@bot.message_handler(func=lambda message: message.reply_to_message and message.chat.id in user_support_chats and 
                                         not user_support_chats[message.chat.id].get("in_support_mode", False) and
                                         message.reply_to_message.message_id == user_support_chats[message.chat.id].get("prompt_message_id"))
def handle_support_request(message):
    chat_id = message.chat.id
    user_support_chats[chat_id]["in_support_mode"] = True
    user_support_chats[chat_id]["time_created"] = time.time()
    
    # Check if the support channel is properly setup
    if not support_channel_properly_setup:
        # Fall back to direct messaging with admin
        try:
            user_info = user_support_chats[chat_id]["user_info"]
            support_message = f"{user_info}\n\n<b>Запрос:</b>\n{message.text}"
            
            # Send message directly to admin
            bot.send_message(
                ADMIN_CHAT_ID,
                f"📩 <b>Новый запрос в поддержку</b> (канал поддержки не настроен):\n\n{support_message}\n\n" +
                f"Для ответа используйте: <code>{chat_id}: Ваш ответ</code>",
                parse_mode="HTML"
            )
            
            # Notify the user
            bot.reply_to(
                message,
                f"✅ Ваш запрос отправлен в техническую поддержку. Ожидайте ответа специалиста.\n\n<b>Номер тикета:</b> DM-{int(time.time())}",
                parse_mode="HTML"
            )
            
            # No topic in this case
            ticket_id = f"DM-{int(time.time())}"
            user_support_chats[chat_id]["topic_message_id"] = None
            user_support_chats[chat_id]["topic_name"] = f"Прямой запрос в поддержку {ticket_id}"
            user_support_chats[chat_id]["direct_ticket_id"] = ticket_id
            
            # Save updated support data
            save_support_data()
            
            return
        except Exception as e:
            print(f"Error sending direct support message: {str(e)}")
            bot.reply_to(message, f"❌ Произошла ошибка при создании запроса в поддержку. Пожалуйста, попробуйте позже.")
            user_support_chats.pop(chat_id, None)
            
            # Save updated support data
            save_support_data()
            
            return
    
    # Create a topic in the support channel
    try:
        # Create support message with user information and their request
        user_info = user_support_chats[chat_id]["user_info"]
        support_message = f"{user_info}\n\n<b>Запрос:</b>\n{message.text}"
        
        # Generate a unique topic name
        # topic_name = f"{user_support_chats[chat_id]['first_name']} {int(time.time())}"
        
        # Available Telegram topic colors (RGB format)
        topic_colors = [
            7322096,   # Blue (0x6FB9F0)
            16766590,  # Yellow (0xFFD67E)
            13338331,  # Purple (0xCB86DB)
            9367192,   # Green (0x8EEE98)
            16749490,  # Pink (0xFF93B2)
            16478047   # Red (0xFB6F5F)
        ]
        
        # Choose a random color for the topic
        icon_color = random.choice(topic_colors)
        
        # Create a forum topic using direct API request
        api_url = f"https://api.telegram.org/bot{API_TOKEN}/createForumTopic"
        payload = {
            "chat_id": SUPPORT_CHANNEL_ID,
            "name": f"{user_support_chats[chat_id]['first_name']}",  # Temporary name, will update after getting ID
            "icon_color": icon_color
        }
        
        response = requests.post(api_url, json=payload)
        
        # Extract the message_thread_id from the response
        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                topic_data = result.get("result", {})
                message_thread_id = topic_data.get("message_thread_id")
                
                # Update the topic name with the ticket number
                topic_name = f"{user_support_chats[chat_id]['first_name']} | Номер тикета: {message_thread_id}"
                
                # Update the topic name
                edit_api_url = f"https://api.telegram.org/bot{API_TOKEN}/editForumTopic"
                edit_payload = {
                    "chat_id": SUPPORT_CHANNEL_ID,
                    "message_thread_id": message_thread_id,
                    "name": topic_name
                }
                
                try:
                    requests.post(edit_api_url, json=edit_payload)
                except Exception as e:
                    print(f"Error updating topic name: {str(e)}")
                
                # Now send the actual support request in the newly created topic
                topic_msg = bot.send_message(
                    SUPPORT_CHANNEL_ID,
                    support_message,
                    parse_mode="HTML",
                    message_thread_id=message_thread_id
                )
                
                # Store the topic message ID for future reference
                user_support_chats[chat_id]["topic_message_id"] = message_thread_id
                user_support_chats[chat_id]["topic_name"] = topic_name
                
                # Notify the user that their request has been sent with ticket number
                bot.reply_to(
                    message,
                    f"✅ Ваш запрос отправлен в техническую поддержку. Ожидайте ответа специалиста.\n\n<b>Номер тикета:</b> {message_thread_id}",
                    parse_mode="HTML",
                    reply_markup=get_main_menu_keyboard()
                )
                
                # Set user to initial state
                user_states[chat_id] = "initial"
                
                # Store the first message ID from the user for context
                user_support_chats[chat_id]["first_message_id"] = message.message_id
                
                # Save updated support data
                save_support_data()
            else:
                raise Exception(f"Failed to create topic: {result.get('description', 'Unknown error')}")
        else:
            raise Exception(f"API request failed with status code {response.status_code}")
        
    except Exception as e:
        print(f"Error creating support ticket: {str(e)}")
        bot.reply_to(message, f"❌ Произошла ошибка при создании запроса в поддержку. Пожалуйста, попробуйте позже.")
        user_support_chats.pop(chat_id, None)
        
        # Save updated support data
        save_support_data()

# Handle '/active' command to list all active support chats - THIS MUST COME BEFORE OTHER CHANNEL HANDLERS
@bot.message_handler(commands=['active'])
def list_active_chats(message):
    # Only process in the support channel
    if message.chat.id != SUPPORT_CHANNEL_ID:
        return
        
    if len(user_support_chats) == 0:
        bot.send_message(
            SUPPORT_CHANNEL_ID,
            "🔍 <i>Активных запросов в поддержке не найдено</i>",
            parse_mode="HTML"
        )
        return
    
    # Create a message with all active support chats
    active_chats = "<b>Активные запросы в поддержке:</b>\n\n"
    
    # Get channel ID without the -100 prefix for generating links
    channel_id_for_link = str(SUPPORT_CHANNEL_ID)[4:] if str(SUPPORT_CHANNEL_ID).startswith('-100') else str(SUPPORT_CHANNEL_ID)
    
    for user_id, data in user_support_chats.items():
        first_name = data.get('first_name', 'Неизвестно')
        last_name = data.get('last_name', '')
        username = data.get('username', 'нет')
        topic_id = data.get('topic_message_id', 'нет')
        topic_name = data.get('topic_name', 'Без темы')
        creation_time = data.get('time_created', time.time())
        
        # Format the creation time
        time_str = time.strftime("%d.%m.%Y %H:%M", time.localtime(creation_time))
        
        # Format the elapsed time since creation
        elapsed = time.time() - creation_time
        if elapsed < 60:
            elapsed_str = f"{int(elapsed)} сек"
        elif elapsed < 3600:
            elapsed_str = f"{int(elapsed / 60)} мин"
        elif elapsed < 86400:
            elapsed_str = f"{int(elapsed / 3600)} час"
        else:
            elapsed_str = f"{int(elapsed / 86400)} дн"
        
        active_chats += (
            f"<b>{topic_name}</b>\n"
            f"<b>Пользователь:</b> {first_name} {last_name} "
            f"(@{username})\n"
            f"<b>ID:</b> <code>{user_id}</code>\n"
            f"<b>Создан:</b> {time_str} ({elapsed_str} назад)\n"
        )
        
        # Add topic link if available
        if str(topic_id).isdigit():
            active_chats += f"📝 <b>Тема:</b> <a href='https://t.me/c/{channel_id_for_link}/{topic_id}'>Перейти к обсуждению</a>\n"
        
        # Add quick commands
        active_chats += (
            f"📤 <b>Быстрый ответ:</b> <code>{user_id}: Ваш ответ</code>\n"
        )
    
    bot.send_message(
        SUPPORT_CHANNEL_ID,
        active_chats,
        parse_mode="HTML",
        disable_web_page_preview=True
    )

# Handle replies in support channel threads (lower priority than specific command handlers)
@bot.message_handler(func=lambda message: message.chat.id == SUPPORT_CHANNEL_ID and not message.from_user.is_bot, 
                    content_types=['text', 'photo', 'video', 'audio', 'document', 'voice', 'sticker'])
def handle_support_reply(message):
    if message.content_type == 'text' and message.text and message.text.startswith('/'):
        return
    
    # For topic messages
    if message.is_topic_message:
        # Find the user associated with this topic
        user_id = None
        topic_message_id = message.message_thread_id
        
        for uid, data in user_support_chats.items():
            if data.get("topic_message_id") == topic_message_id:
                user_id = uid
                break
        
        if user_id:
            try:
                # Extract the text part only for non-media messages
                text = ""
                if message.content_type == 'text':
                    text = message.text
                elif hasattr(message, 'caption') and message.caption:
                    text = message.caption
                else:
                    # If no text, send a generic message about media
                    text = "Сообщение от поддержки (содержит медиа-файл)"
                
                # Create a simple inline keyboard with a Reply button
                reply_keyboard = InlineKeyboardMarkup(row_width=1)
                reply_keyboard.add(InlineKeyboardButton("✍️ Ответить на сообщение", callback_data=f"reply_{topic_message_id}"))
                
                # Send reply text to user if present
                if text:
                    #print(f"Sending text to user {user_id}: {text[:30]}...")
                    # Add notification about new support message
                    support_message = f"<b>Ответ поддержки</b> (тикет #{topic_message_id}):\n\n{text}"
                    
                    # Send message with Reply button
                    bot.send_message(
                        user_id,
                        support_message,
                        parse_mode="HTML",
                        reply_markup=reply_keyboard,
                        disable_web_page_preview=True
                    )
                    
                    # No longer sending "Выберите действие" messages and extra keyboards
                
                # Forward the actual media content based on type
                if message.content_type == 'photo':
                    photo = message.photo[-1]  # Get largest photo
                    #print(f"Sending photo to user {user_id} with file_id: {photo.file_id}")
                    caption = message.caption or "<b>Фото от поддержки</b>"
                    bot.send_photo(
                        user_id, 
                        photo.file_id, 
                        caption=caption,
                        parse_mode="HTML",
                        reply_markup=reply_keyboard
                    )
                elif message.content_type == 'document':
                    #print(f"Sending document to user {user_id} with file_id: {message.document.file_id}")
                    caption = message.caption or "<b>Документ от поддержки</b>"
                    bot.send_document(
                        user_id, 
                        message.document.file_id, 
                        caption=caption,
                        parse_mode="HTML",
                        reply_markup=reply_keyboard
                    )
                elif message.content_type == 'video':
                    #print(f"Sending video to user {user_id} with file_id: {message.video.file_id}")
                    caption = message.caption or "<b>Видео от поддержки</b>"
                    bot.send_video(
                        user_id, 
                        message.video.file_id, 
                        caption=caption,
                        parse_mode="HTML",
                        reply_markup=reply_keyboard
                    )
                elif message.content_type == 'audio':
                    caption = message.caption or "<b>Аудио от поддержки</b>"
                    #print(f"Sending audio to user {user_id} with file_id: {message.audio.file_id}")
                    bot.send_audio(
                        user_id, 
                        message.audio.file_id, 
                        caption=caption, 
                        parse_mode="HTML", 
                        reply_markup=reply_keyboard
                    )
                elif message.content_type == 'voice':
                    caption = message.caption or "<b>Голосовое сообщение от поддержки</b>"
                    #print(f"Sending voice to user {user_id} with file_id: {message.voice.file_id}")
                    bot.send_voice(
                        user_id, 
                        message.voice.file_id, 
                        caption=caption, 
                        parse_mode="HTML", 
                        reply_markup=reply_keyboard
                    )
                elif message.content_type == 'sticker':
                    #print(f"Sending sticker to user {user_id} with file_id: {message.sticker.file_id}")
                    bot.send_sticker(
                        user_id, 
                        message.sticker.file_id
                    )
                    # Also send the Reply button after the sticker
                    bot.send_message(
                        user_id,
                        "<b>Стикер от поддержки</b>",
                        parse_mode="HTML",
                        reply_markup=reply_keyboard
                    )
                
                # Schedule auto-close if the user doesn't respond
                closer_name = f"{message.from_user.first_name}"
                if message.from_user.last_name:
                    closer_name += f" {message.from_user.last_name}"
                
                # Use threading to schedule the auto-close
                import threading
                def auto_close():
                    # Check if the ticket is still open
                    if user_id in user_support_chats and user_support_chats[user_id].get("topic_message_id") == topic_message_id:
                        close_support_ticket(topic_message_id, f"автоматически после ответа поддержки")
                
                # Start the timer
                timer = threading.Timer(86400.0, auto_close)
                timer.daemon = True  # Allow the thread to exit when the main program exits
                timer.start()
                
            except Exception as e:
                print(f"Error sending support reply: {str(e)}")
                bot.send_message(
                    SUPPORT_CHANNEL_ID,
                    f"❌ <i>Ошибка отправки сообщения пользователю: {str(e)}</i>",
                    parse_mode="HTML",
                    message_thread_id=topic_message_id
                )
        else:
            print(f"No user found for topic {topic_message_id}")
            bot.send_message(
                SUPPORT_CHANNEL_ID,
                "❌ <i>Не удалось найти пользователя. Возможно, тикет устарел или пользователь отменил запрос.</i>",
                parse_mode="HTML",
                message_thread_id=topic_message_id
            )
    # For direct messages in the channel (not in topics)
    else:
        # Only process text messages for direct replies (format: user_id: text)
        if message.content_type != 'text' or not message.text:
            bot.send_message(
                SUPPORT_CHANNEL_ID,
                "❌ <i>Для прямой отправки сообщений пользователю используйте текстовый формат: [ID пользователя]: [текст сообщения]</i>",
                parse_mode="HTML"
            )
            return
            
        # Try to find a user ID in the format: 123456789: message
        try:
            if ':' in message.text:
                parts = message.text.split(':', 1)
                potential_id = parts[0].strip()
                reply_text = parts[1].strip()
                
                if potential_id.isdigit():
                    user_id = int(potential_id)
                    
                    # Check if this user exists in our support chats
                    user_exists = user_id in user_support_chats
                    
                    # Get ticket ID
                    ticket_id = None
                    if user_exists:
                        topic_msg_id = user_support_chats[user_id].get("topic_message_id")
                        if topic_msg_id:
                            ticket_id = topic_msg_id
                        else:
                            ticket_id = user_support_chats[user_id].get("direct_ticket_id", "Unknown")
                    else:
                        ticket_id = "Unknown"
                    
                    # Send message to the user
                    bot.send_message(
                        user_id,
                        f"<b>Ответ поддержки</b> (тикет #{ticket_id}):\n\n{reply_text}",
                        parse_mode="HTML"
                    )
                    
                else:
                    bot.send_message(
                        SUPPORT_CHANNEL_ID,
                        "❌ <i>Неверный формат. Используйте: [ID пользователя]: [текст сообщения]</i>",
                        parse_mode="HTML"
                    )
            else:
                bot.send_message(
                    SUPPORT_CHANNEL_ID,
                    "❌ <i>Неверный формат. Используйте: [ID пользователя]: [текст сообщения]</i>",
                    parse_mode="HTML"
                )
        except Exception as e:
            print(f"Error sending direct message: {str(e)}")
            bot.send_message(
                SUPPORT_CHANNEL_ID,
                f"❌ <i>Ошибка отправки сообщения: {str(e)}</i>",
                parse_mode="HTML"
            )

# Handle '/documents' command to list all available documents
@bot.message_handler(commands=['documents'])
def list_documents(message):
    global available_documents
    document_files = sorted(glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf")))
    if not document_files:
        bot.reply_to(message, "PDF документы не найдены в директории documents.")
        return
    
    available_documents = document_files
    
    doc_list = "📚 <b>Доступные документы</b>:\n\n"
    
    for i, doc_path in enumerate(document_files):
        doc_name = os.path.basename(doc_path)
        file_size_mb = round(os.path.getsize(doc_path) / (1024 * 1024), 1)
        doc_list += f"<b>{i+1}.</b> {doc_name} ({file_size_mb} МБ)\n"
    
    doc_list += "\n<i>Для скачивания просто отправьте номер документа в чат</i>"
    
    # Set user state to documents mode so we can handle numeric responses
    user_states[message.chat.id] = "documents"
    
    # In testing mode, don't show cancel button
    if TESTING_MODE:
        # Send the message with no keyboard
        bot.send_message(message.chat.id, doc_list, parse_mode="HTML", reply_markup=telebot.types.ReplyKeyboardRemove())
    else:
        # Send the message with cancel button
        bot.send_message(message.chat.id, doc_list, parse_mode="HTML", reply_markup=get_cancel_keyboard())

# Handle simple numeric responses for document download
@bot.message_handler(func=lambda message: message.text.isdigit())
def handle_document_number(message):
    try:
        doc_number = int(message.text)
        
        # Check if document list is available
        if not available_documents:
            # Reload document list if empty
            document_files = sorted(glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf")))
            if not document_files:
                bot.reply_to(message, "PDF документы не найдены. Используйте /documents для просмотра списка документов.")
                return
            available_documents.extend(document_files)
        
        # Check if the number is valid
        if doc_number < 1 or doc_number > len(available_documents):
            bot.reply_to(message, f"Введите номер от 1 до {len(available_documents)}.")
            return
        
        # Get the document path
        doc_path = available_documents[doc_number - 1]
        doc_name = os.path.basename(doc_path)
        
        progress_msg = bot.reply_to(message, f"⏳ Отправка документа '{doc_name}'...")
        
        # Send the document without caption
        with open(doc_path, 'rb') as doc_file:
            bot.send_document(
                message.chat.id, 
                doc_file
            )
        
        # Delete progress message
        try:
            bot.delete_message(chat_id=message.chat.id, message_id=progress_msg.message_id)
        except Exception as e:
            print(f"Error deleting message: {str(e)}")
        
        if TESTING_MODE:
            # In testing mode, return to normal question answering mode without buttons
            user_states[message.chat.id] = "normal"
            bot.send_message(
                message.chat.id,
                "Документ отправлен. Вы можете продолжить задавать вопросы.",
                reply_markup=telebot.types.ReplyKeyboardRemove()
            )
        else:
            # In normal mode, stay in documents mode with Cancel button
            bot.send_message(
                message.chat.id,
                "Документ отправлен. Вы можете выбрать другой документ или нажать 'Отмена' для возврата в главное меню.",
                reply_markup=get_cancel_keyboard()
            )
            
    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {str(e)}")

# Process all incoming messages based on user mode
@bot.message_handler(func=lambda message: True, content_types=['text', 'photo', 'video', 'audio', 'document', 'voice', 'sticker'])
def process_message(message):
    chat_id = message.chat.id
    
    # Get current user state
    current_state = user_states.get(chat_id, "initial")
    
    # Handle document number input in testing mode and documents state
    if TESTING_MODE and message.content_type == 'text' and message.text.isdigit() and current_state == "documents":
        handle_document_number(message)
        return
    
    # In testing mode, treat all other text messages as RAG queries
    if TESTING_MODE and message.content_type == 'text' and not message.text.startswith('/'):
        process_with_rag(message)
        return
    
    # Check if the user has an active ticket
    has_active_ticket = chat_id in user_support_chats and user_support_chats[chat_id].get("in_support_mode", False)
    
    # Handle keyboard menu selections first
    if message.content_type == 'text':
        if message.text == "💬 Задать вопрос ИИ" or message.text == "💬 Вернуться к вопросам ИИ" or message.text == "💬 Продолжить с ИИ":
            user_states[chat_id] = "normal"
            bot.send_message(
                chat_id,
                "Задайте любой вопрос, по базе данных документов",
                reply_markup=get_cancel_keyboard()
            )
            return
        elif message.text == "❌ Отмена":
            user_states[chat_id] = "initial"
            bot.send_message(
                chat_id,
                "Вы вернулись в главное меню. Выберите режим работы:",
                reply_markup=get_main_menu_keyboard()
            )
            return
        elif message.text == "📞 Поддержка ЧатБота":
            start_support_chat(message)
            return
        elif message.text == "📚 Документы":
            user_states[chat_id] = "documents"

            # Show available documents
            document_files = sorted(glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf")))
            if not document_files:
                bot.send_message(
                    chat_id,
                    "PDF документы не найдены в директории documents.", 
                    reply_markup=get_main_menu_keyboard()
                )
                return
            
            # Update global list of available documents
            global available_documents
            available_documents = document_files
            
            doc_list = "📚 <b>Доступные документы</b>:\n\n"
            
            for i, doc_path in enumerate(document_files):
                doc_name = os.path.basename(doc_path)
                file_size_mb = round(os.path.getsize(doc_path) / (1024 * 1024), 1)
                doc_list += f"<b>{i+1}.</b> {doc_name} ({file_size_mb} МБ)\n"
            
            doc_list += "\n<i>Для скачивания просто отправьте номер документа в чат</i>"       
            bot.send_message(
                chat_id, 
                doc_list, 
                parse_mode="HTML", 
                reply_markup=get_cancel_keyboard()
            )
            return
        elif message.text == "📝 Создать обращение в поддержку":
            # Check if user already has an active ticket
            if has_active_ticket:
                bot.send_message(
                    chat_id,
                    "У вас уже есть активный запрос в поддержку. Пожалуйста, дождитесь ответа или закройте текущий запрос перед созданием нового.",
                    reply_markup=get_support_menu_keyboard(True)
                )
                return
            create_new_support_request(message)
            return
        elif message.text == "👋 Выйти на главное меню":
            # Just change mode without closing tickets
            user_states[chat_id] = "normal"
            bot.send_message(
                chat_id,
                "Вы вышли из режима поддержки. Ваши активные обращения остаются открытыми, и вы получите уведомление при ответе от поддержки.",
                reply_markup=get_main_menu_keyboard()
            )
            return
        elif message.text == "🎫 Мое активное обращение":
            view_active_ticket(message)
            return
        elif message.text == "❌ Закрыть тикет":
            if chat_id in user_support_chats and user_support_chats[chat_id].get("in_support_mode", False):
                topic_msg_id = user_support_chats[chat_id].get("topic_message_id")
                if topic_msg_id:
                    try:
                        # Mark ticket as closed in support channel
                        close_message = "🔴 <b>Тикет закрыт пользователем</b>"
                        bot.send_message(
                            SUPPORT_CHANNEL_ID,
                            close_message,
                            parse_mode="HTML",
                            message_thread_id=topic_msg_id
                        )
                        
                        # Notify user and show main menu
                        bot.send_message(
                            chat_id,
                            f"Тикет #{topic_msg_id} закрыт. Если у вас возникнут новые вопросы, создайте новый запрос.",
                            reply_markup=get_main_menu_keyboard()
                        )
                        
                        # Delete the ticket from the JSON file
                        delete_ticket_from_json(topic_msg_id)
                        
                        # Remove ticket and set state to normal
                        user_support_chats.pop(chat_id, None)
                        user_states[chat_id] = "normal"
                    except Exception as e:
                        print(f"Error closing ticket: {str(e)}")
                        bot.reply_to(
                            message, 
                            "Не удалось закрыть тикет. Попробуйте позже.",
                            reply_markup=get_support_menu_keyboard(True)
                        )
                else:
                    bot.reply_to(
                        message, 
                        "Не удалось найти активный тикет.",
                        reply_markup=get_main_menu_keyboard()
                    )
                    user_states[chat_id] = "normal"
            else:
                bot.reply_to(
                    message, 
                    "У вас нет активных тикетов.",
                    reply_markup=get_main_menu_keyboard()
                )
                user_states[chat_id] = "normal"
            return
    
    # Handle numeric input in documents mode
    if current_state == "documents" and message.content_type == 'text' and message.text.isdigit():
        try:
            doc_number = int(message.text)
            
            # Check if document list is available
            if not available_documents:
                # Reload document list if empty
                document_files = sorted(glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf")))
                if not document_files:
                    bot.reply_to(
                        message, 
                        "PDF документы не найдены. Используйте кнопку 'Документы' для просмотра списка документов.", 
                        reply_markup=get_cancel_keyboard()
                    )
                    return
                available_documents.extend(document_files)
            
            # Check if the number is valid
            if doc_number < 1 or doc_number > len(available_documents):
                bot.reply_to(
                    message, 
                    f"Введите номер от 1 до {len(available_documents)}.", 
                    reply_markup=get_cancel_keyboard()
                )
                return
            
            # Get the document path
            doc_path = available_documents[doc_number - 1]
            doc_name = os.path.basename(doc_path)
            
            # Send progress message
            progress_msg = bot.reply_to(message, f"⏳ Отправка документа '{doc_name}'...")
            
            # Send the document without caption
            with open(doc_path, 'rb') as doc_file:
                bot.send_document(
                    chat_id, 
                    doc_file
                )
            
            try:
                bot.delete_message(chat_id=chat_id, message_id=progress_msg.message_id)
            except Exception as e:
                print(f"Error deleting message: {str(e)}")

            bot.send_message(
                chat_id,
                "Документ отправлен. Вы можете выбрать другой документ или нажать 'Отмена' для возврата в главное меню.",
                reply_markup=get_cancel_keyboard()
            )
                
        except Exception as e:
            bot.reply_to(
                message, 
                f"❌ Ошибка: {str(e)}", 
                reply_markup=get_cancel_keyboard()
            )
        return
    
    # Handle text input based on state
    if current_state == "initial":
        # In initial state, only allow button clicks, ignore text input
        bot.send_message(
            chat_id,
            "Выберите из опций ниже:",
            reply_markup=get_main_menu_keyboard()
        )
        return
    elif current_state == "selecting_ticket":
        # User is in the process of selecting a ticket, remind to use the inline buttons
        bot.send_message(
            chat_id,
            "Пожалуйста, выберите тикет с помощью кнопок выше.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    elif current_state == "documents":
        # If in documents mode but input is not a number, remind to enter a number
        if message.content_type == 'text':
            bot.send_message(
                chat_id,
                "Введите номер документа для скачивания или нажмите 'Отмена'.",
                reply_markup=get_cancel_keyboard()
            )
        else:
            bot.send_message(
                chat_id,
                "Для скачивания документа введите его номер.",
                reply_markup=get_cancel_keyboard()
            )
        return
    
    # Debug information about the message
    print(f"Received message type: {message.content_type} from user ID: {chat_id}")
    
    # If this is a message in the support channel from an admin
    if chat_id == SUPPORT_CHANNEL_ID:
        return
    
    # Handle based on current state
    if current_state == "support" and chat_id in user_support_chats and user_support_chats[chat_id].get("in_support_mode", True):
        # Forward user's message to the support channel thread
        topic_msg_id = user_support_chats[chat_id].get("topic_message_id")
        
        if topic_msg_id:
            try:
                if message.content_type == 'text':
                    bot.send_message(
                        SUPPORT_CHANNEL_ID,
                        f"<b>Сообщение от пользователя:</b>\n{message.text}",
                        parse_mode="HTML",
                        message_thread_id=topic_msg_id
                    )
                # Forward media content if present
                elif message.content_type == 'photo':
                    photo = message.photo[-1]  # Get largest photo
                    caption = message.caption or "<b>Фото от пользователя</b>"

                    bot.send_photo(
                        SUPPORT_CHANNEL_ID, 
                        photo.file_id, 
                        caption=caption,
                        parse_mode="HTML",
                        message_thread_id=topic_msg_id
                    )
                elif message.content_type == 'document':
                    caption = message.caption or "<b>Документ от пользователя</b>"
                    #print(f"Forwarding document with file_id: {message.document.file_id}")
                    bot.send_document(
                        SUPPORT_CHANNEL_ID, 
                        message.document.file_id, 
                        caption=caption,
                        parse_mode="HTML",
                        message_thread_id=topic_msg_id
                    )
                elif message.content_type == 'video':
                    caption = message.caption or "<b>Видео от пользователя</b>"
                    bot.send_video(
                        SUPPORT_CHANNEL_ID, 
                        message.video.file_id, 
                        caption=caption,
                        parse_mode="HTML",
                        message_thread_id=topic_msg_id
                    )
                elif message.content_type == 'audio':
                    caption = message.caption or "<b>Аудио от пользователя</b>"
                    #print(f"Forwarding audio with file_id: {message.audio.file_id}")
                    bot.send_audio(
                        SUPPORT_CHANNEL_ID, 
                        message.audio.file_id, 
                        caption=caption,
                        parse_mode="HTML",
                        message_thread_id=topic_msg_id
                    )
                elif message.content_type == 'voice':
                    caption = message.caption or "<b>Голосовое сообщение от пользователя</b>"
                    #print(f"Forwarding voice with file_id: {message.voice.file_id}")
                    bot.send_voice(
                        SUPPORT_CHANNEL_ID, 
                        message.voice.file_id, 
                        caption=caption,
                        parse_mode="HTML",
                        message_thread_id=topic_msg_id
                    )
                elif message.content_type == 'sticker':
                    #print(f"Forwarding sticker with file_id: {message.sticker.file_id}")
                    bot.send_sticker(
                        SUPPORT_CHANNEL_ID,
                        message.sticker.file_id,
                        message_thread_id=topic_msg_id
                    )
                    # Also send a notification about the sticker
                    bot.send_message(
                        SUPPORT_CHANNEL_ID,
                        "<b>Пользователь отправил стикер</b>",
                        parse_mode="HTML",
                        message_thread_id=topic_msg_id
                    )
                else:
                    #print(f"Received unsupported content type: {message.content_type}")
                    bot.send_message(
                        SUPPORT_CHANNEL_ID,
                        f"<b>Пользователь отправил сообщение типа:</b> {message.content_type}",
                        parse_mode="HTML",
                        message_thread_id=topic_msg_id
                    )
                
                # After the forwarding is complete, always show the support menu again
                bot.reply_to(message, "✅ Ваше сообщение отправлено в поддержку", reply_markup=get_support_menu_keyboard(True))
            except Exception as e:
                print(f"Error forwarding message to support: {str(e)}")
                bot.reply_to(message, "❌ Не удалось отправить сообщение в поддержку. Пожалуйста, попробуйте позже.", reply_markup=get_support_menu_keyboard(True))
        else:
            bot.reply_to(message, "❌ Ошибка в системе поддержки. Пожалуйста, создайте новый запрос с помощью /support", reply_markup=get_main_menu_keyboard())
            user_states[chat_id] = "normal"
    else:
        # Only process text messages with RAG if in normal mode
        if current_state == "normal":
            if message.content_type == 'text':
                process_with_rag(message)
            else:
                bot.reply_to(message, "Я могу отвечать только на текстовые вопросы. Пожалуйста, задайте ваш вопрос текстом.", reply_markup=get_cancel_keyboard())
        else:
            user_states[chat_id] = "initial"
            bot.send_message(
                chat_id,
                "Выберите из опций ниже:",
                reply_markup=get_main_menu_keyboard()
            )

# Preprocess user query to filter out document names that can add noise to RAG
def preprocess_rag_query(query: str) -> str:
    """
    Remove document names and variants from the query to reduce noise in RAG results.
    """
    # Document name patterns to filter out (case insensitive)
    doc_patterns = [
            r'\bДОПОГ\w*\b', r'\bДопог\w*\b', r'\bдопог\w*\b',
    ]
    
    query_stripped = query.strip()
    
    is_only_doc_name = False
    for pattern in doc_patterns:
        if re.match(f"^{pattern}$", query_stripped, re.IGNORECASE):
            is_only_doc_name = True
            break
    
    if is_only_doc_name:
        return query_stripped
    
    processed_query = query
    for pattern in doc_patterns:
        processed_query = re.sub(pattern, '', processed_query, flags=re.IGNORECASE)
    
    processed_query = ' '.join(processed_query.split())
    
    if not processed_query.strip():
        return query
    
    return processed_query

# RAG processing function
def process_with_rag(message):
    user_query = message.text
    user_id = message.from_user.id
    username = message.from_user.username or ""
    timestamp = datetime.datetime.now().isoformat()
    start_time = time.time()
    
    user_language_code = message.from_user.language_code if hasattr(message.from_user, "language_code") else "unknown"
    
    message_id = message.message_id
    
    is_premium = message.from_user.is_premium if hasattr(message.from_user, "is_premium") else False
    
    log_data = {
        "user_id": user_id,
        "username": username,
        "query": user_query,
        "timestamp": timestamp,
        "message_id": message_id,
        "processing_times": {},
        "rag_results": [],
        "llm_response": "",
        "user_meta": {
            "language_code": user_language_code,
            "is_premium": is_premium
        },
        "feedback": 0
    }
    
    processing_msg = bot.reply_to(message, "⏳ ИИ обрабатывает ваш запрос. Подождите. . .")
    
    try:
        # Preprocess query to remove document names (reduces noise in RAG search)
        processed_query = preprocess_rag_query(user_query)
        
        # Log the original and processed queries
        log_data["original_query"] = user_query
        log_data["processed_query"] = processed_query
        
        # Generate embedding for the preprocessed query
        embed_start = time.time()
        query_vector = encoder.encode(processed_query).tolist()
        embed_time = time.time() - embed_start
        log_data["processing_times"]["embedding"] = embed_time
        
        # Query Qdrant for relevant documents
        query_start = time.time()
        hits = qdrant_client.search(
            collection_name="docs",
            query_vector=query_vector,
            score_threshold = 0.001,
            limit=20
        )
        query_time = time.time() - query_start
        log_data["processing_times"]["rag_query"] = query_time
        
        # Prepare context from search results
        result = []
        for hit in hits:
            result.append({
                "payload": hit.payload,
                "score": hit.score
            })
        
        result = remove_duplicate_search_results(result)
        
        original_deduped_count = len(result)
        result = result[:10]
        config_map = {
            "ПДД.pdf": (0, 1),
            "ГОСТ-Р-57478-2017.pdf": (2, 1),
            "Европейское_соглашение_о_международной_дорожной_перевозке_опасных.pdf": (1, 1),
            "ДОПОГ_том_1.pdf": (26, 1),
            "ДОПОГ_том_2.pdf": (16, 1)
        }
        
        for ctx in result:
            source = ctx['payload']['metadata'].get('source', '')
            source_basename = os.path.basename(source) if isinstance(source, str) else ''
            
            page_offset = 0
            for key, (_, offset) in config_map.items():
                if key.lower() in source_basename.lower() or any(k.lower() in source_basename.lower() for k in key.split('-')):
                    page_offset = offset
                    break
            
            if page_offset != 0 and 'page' in ctx['payload']['metadata']:
                ctx['payload']['metadata']['page'] += page_offset
        
        # Save RAG results for logging with applied page offsets
        for idx, ctx in enumerate(result):
            source = ctx['payload']['metadata'].get('source', 'Unknown')
            source = os.path.basename(source) if isinstance(source, str) else 'Unknown'
            
            # The page number already has the offset applied above
            page = ctx['payload']['metadata'].get('page', 'N/A')
            
            log_data["rag_results"].append({
                "content": ctx['payload']['page_content'],
                "source": source,
                "page": page,
                "score": ctx['score']
            })
        
        context_str = "\n".join(
            f"Текст: {ctx['payload']['page_content']}\n"
            f"Страница: {ctx['payload']['metadata']['page']}, "
            f"Источник: {ctx['payload']['metadata']['source']}"
            for ctx in result
        )
        messages = [
            {
                "role": "system",
                "text": """
                Ты — дружелюбный и разговорчивый помощник который отвечает на вопросы водителя фуры по документам. 
                Отвечай на вопросы пользователя подробно и в непринужденном разговорном стиле. 
                Используй информацию из базы знаний как основу, но не ограничивайся сухими фактами — дополняй ответ интересными деталями, примерами и собственными рассуждениями. 
                Можешь использовать разговорные выражения, юмор и эмоции, чтобы ответ звучал естественно и увлекательно. 
                При этом важно, чтобы основная суть ответа соответствовала фактам из базы знаний. 
                Старайся создать ощущение живого диалога с реальным человеком.
                
                Если в предоставленном контексте нет информации для ответа на вопрос, но тема связана с работой водителя грузовика, логистикой или документацией - ОБЯЗАТЕЛЬНО дай ответ на основе своих общих знаний, ОБЯЗАТЕЛЬНО В таком случае честно полностью напиши данную строку, что "Точной информации по этому вопросу нет в загруженных во мне документах, я могу поделиться общими сведениями, но их стоит перепроверить..." и продолжи отвечать своими словами.
                
                Когда используешь источник из базы знаний, всегда отвечай точно и ссылайся на источник только который ты использовал в формате:
                Источник: {source}, Страница: {page}, Пункт: {point}
                Не пиши путь до файла, только название файла в формате: "Документ.pdf"
                Если пункт не указан, пиши что "Пункт: не указан"
                Пиши источник под ответом (абзацем).

                Если информация из разных источников — перечисли все ссылки.
                
                ВАЖНО: Никогда не отвечай "Я не могу найти информацию в базе знаний" или "В предоставленном контексте нет ответа". Всегда пытайся дать полезный ответ по теме.
                """,
            },
            {
                "role": "user",
                "text": f"""Вопрос: {user_query}

Контекст:
{context_str}

Ответь на вопрос, используя указанный контекст. Если в контексте нет точного ответа, но вопрос связан с темой водителей грузовиков или логистики, дай ответ из своих знаний. Укажи все релевантные источники, если они есть.""",
            },
        ]
        
        # Process with Yandex GPT
        llm_start = time.time()
        completion_result = yandex_sdk.models.completions(model_name="yandexgpt", model_version="rc").configure(temperature=0.5).run(messages)
        llm_time = time.time() - llm_start
        log_data["processing_times"]["llm"] = llm_time
        
        llm_response = ""
        for alternative in completion_result:
            llm_response = alternative.text
            break
        
        # Save LLM response to log
        log_data["llm_response"] = llm_response
        
        try:
            bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)
        except Exception as e:
            print(f"Error deleting message: {str(e)}")
        
        # Format LLM response - Escape any HTML content in the response
        # This prevents HTML parsing issues from the LLM output
        safe_response = llm_response.replace('<', '&lt;').replace('>', '&gt;')
        llm_response_formatted = f"<b>⚡️ Ответ ИИ:</b>\n\n{safe_response}"
        
        # Format RAG results - ограничим до 5 результатов для Telegram-сообщения (чтобы избежать ошибки "message is too long")
        rag_results = "\n\n\n🔍 <b>Найденная информация в документах:</b>\n\n"
        display_results = result[:5]
        
        for i, ctx in enumerate(display_results):
            source = ctx['payload']['metadata'].get('source', 'Неизвестный документ')
            source = os.path.basename(source) if isinstance(source, str) else 'Неизвестный документ'
            page = ctx['payload']['metadata'].get('page', 'N/A')
            
            # Calculate relevance
            similarity = f" (релевантность: {ctx['score']*100:.1f}%)"
            
            # Source info in italics - Fix: Convert page to string and escape any HTML characters
            # HTML special characters need to be escaped to prevent parsing issues
            page_str = str(page).replace('<', '&lt;').replace('>', '&gt;')
            source_str = str(source).replace('<', '&lt;').replace('>', '&gt;')
            source_info = f"📁 <i>Источник: {source_str}, страница: {page_str}</i>"
            
            # Get a snippet of the content (first 100 chars instead of 150)
            content = ctx['payload']['page_content']
            safe_content = content.replace('<', '&lt;').replace('>', '&gt;')
            snippet = safe_content[:150] + "..." if len(safe_content) > 150 else safe_content
            
            # Add blockquote to RAG results
            rag_results += f"<blockquote>{i+1}. {snippet}{similarity}\n{source_info}</blockquote>\n\n"
        

        
        # Combine and send the response - LLM first, then RAG
        combined_response = f"{llm_response_formatted}{rag_results}"
        
        feedback_keyboard = InlineKeyboardMarkup(row_width=2)
        feedback_keyboard.add(
            InlineKeyboardButton("👍 Выдача была полезной", callback_data=f"feedback_good_{message_id}"),
            InlineKeyboardButton("👎 Выдача не ответила на мой вопрос", callback_data=f"feedback_bad_{message_id}")
        )
        
        try:
            bot.reply_to(message, combined_response, parse_mode="HTML", reply_markup=feedback_keyboard)
        except Exception as e:
            if "message is too long" in str(e):
                print("Message too long, sending only LLM response without RAG results")
                bot.reply_to(message, llm_response_formatted, parse_mode="HTML", reply_markup=feedback_keyboard)
            else:
                bot.reply_to(message, f"❌ Ошибка при отправке ответа: {str(e)}", reply_markup=get_cancel_keyboard())
        

        total_time = time.time() - start_time
        log_data["processing_times"]["total"] = total_time
              
        # Save log to file
        save_query_log(log_data)
        
    except Exception as e:
        # In case of error, update the processing message instead of leaving it hanging
        try:
            bot.edit_message_text(
                chat_id=message.chat.id, 
                message_id=processing_msg.message_id,
                text=f"❌ Произошла ошибка при обработке запроса: {str(e)}"
            )
        except Exception as edit_error:
            # If we can't edit the message, try to send a new one
            try:
                bot.send_message(
                    chat_id=message.chat.id,
                    text=f"❌ Произошла ошибка при обработке запроса: {str(e)}",
                    reply_markup=get_cancel_keyboard()
                )
            except:
                pass
        
        # Log error
        print(f"Error processing request: {str(e)}")

        log_data["error"] = str(e)
        log_data["processing_times"]["total"] = time.time() - start_time
        save_query_log(log_data)

# Handle specific commands - register these first for higher priority
@bot.message_handler(commands=['help', 'start', 'support', 'documents', 'cancel', 'active', 'close'])
def handle_commands(message):
    command = message.text.split()[0][1:]  # Extract command without '/'
    
    if TESTING_MODE and command not in ['start', 'help', 'documents']:
        bot.reply_to(message, "В режиме тестирования команды отключены. Просто задавайте вопросы в чат.")
        return
    
    if command in ['help', 'start']:
        send_welcome(message)
    elif command == 'support':
        start_support_chat(message)
    elif command == 'documents':
        list_documents(message)
    elif command == 'cancel':
        cancel_support_chat(message)
    elif command == 'active':
        list_active_chats(message)
    elif command == 'close':
        close_ticket_command(message)

# Function to close a support ticket
def close_support_ticket(topic_id, closer_info, from_support=True):
    # Find the user associated with this topic
    user_id = None
    
    for uid, data in user_support_chats.items():
        if data.get("topic_message_id") == topic_id:
            user_id = uid
            break
    
    if user_id:
        try:
            close_message = f"🔴 <b>Тикет закрыт</b> ({closer_info})"
            bot.send_message(
                SUPPORT_CHANNEL_ID,
                close_message,
                parse_mode="HTML",
                message_thread_id=topic_id
            )
        except Exception as e:
            print(f"Error closing topic: {str(e)}")
        
        bot.send_message(
            user_id,
            f"Ваш запрос в поддержку (тикет #{topic_id}) закрыт. Если у вас возникнут новые вопросы, используйте команду /support.",
            reply_markup=get_main_menu_keyboard()
        )
        
        user_support_chats.pop(user_id, None)
        
        delete_ticket_from_json(topic_id)
        
        return True
    else:
        if from_support:
            bot.send_message(
                SUPPORT_CHANNEL_ID,
                "❌ <i>Не удалось найти пользователя. Возможно, тикет устарел или пользователь отменил запрос.</i>",
                parse_mode="HTML",
                message_thread_id=topic_id
            )
        return False

# Handle '/close' command to close ticket from support side
@bot.message_handler(commands=['close'])
def close_ticket_command(message):
    # Only process in the support channel
    if message.chat.id != SUPPORT_CHANNEL_ID:
        return
        
    if not message.is_topic_message:
        bot.reply_to(message, "❌ Эта команда должна использоваться в треде тикета.")
        return
    
    topic_id = message.message_thread_id
    closer_name = f"{message.from_user.first_name}"
    if message.from_user.last_name:
        closer_name += f" {message.from_user.last_name}"

    if close_support_ticket(topic_id, f"специалист поддержки: {closer_name}"):
        pass
    else:
        pass

# Add handler for the reply button callback
@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call):
    # Обрабатываем callback для обратной связи
    if call.data.startswith('feedback_'):
        try:
            feedback_data = call.data.split('_')
            if len(feedback_data) >= 3:
                feedback_type = feedback_data[1]
                original_message_id = int(feedback_data[2])
                
                # 0 - neutral, 1 - good feedback, 2 - bad feedback
                feedback_value = 1 if feedback_type == "good" else 2
                user_id = call.from_user.id
                
                # Load existing logs
                logs = []
                if os.path.exists(LOGS_FILE):
                    try:
                        with open(LOGS_FILE, 'r', encoding='utf-8') as f:
                            logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
                
                # Find record by message_id
                log_updated = False
                log_idx_to_update = None
                for i, log in enumerate(logs):
                    if log.get("user_id") == user_id and log.get("message_id") == original_message_id:
                        log_idx_to_update = i
                        break
                
                # If record not found by message_id, try to find the last record of this user
                # as a backup (only if original_message_id was not specified in the log)
                if log_idx_to_update is None:
                    latest_log_idx = None
                    latest_timestamp = None
                    
                    for i, log in enumerate(logs):
                        if log.get("user_id") == user_id and not log.get("message_id"):
                            log_timestamp = log.get("timestamp", "")
                            if latest_timestamp is None or log_timestamp > latest_timestamp:
                                latest_timestamp = log_timestamp
                                latest_log_idx = i
                    
                    log_idx_to_update = latest_log_idx
                
                # If record found for update
                if log_idx_to_update is not None:
                    logs[log_idx_to_update]["feedback"] = feedback_value
                    if "message_id" not in logs[log_idx_to_update]:
                        logs[log_idx_to_update]["message_id"] = original_message_id
                    log_updated = True
                else:
                    log_updated = False
                
                # If updated log, save back to file
                if log_updated:
                    with open(LOGS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(logs, f, ensure_ascii=False, indent=2)
                else:
                    print(f"Warning: Could not find matching log for feedback for user {user_id} and message {original_message_id}")
                
                if feedback_type == "good":
                    bot.answer_callback_query(call.id, "Спасибо за положительный отзыв!")
                    try:
                        bot.edit_message_reply_markup(
                            chat_id=call.message.chat.id,
                            message_id=call.message.message_id,
                            reply_markup=None
                        )
                    except Exception as edit_error:
                        print(f"Could not edit message reply markup: {str(edit_error)}")
                    
                    sent_message = bot.send_message(
                        call.message.chat.id,
                        "✅ Спасибо за ваш отзыв! Мы рады, что информация была полезной.",
                        reply_markup=None if TESTING_MODE else get_cancel_keyboard()
                    )
                    
                    # Run function to delete message after 5 seconds
                    def delete_feedback_message():
                        time.sleep(5)
                        try:
                            bot.delete_message(call.message.chat.id, sent_message.message_id)
                        except Exception as e:
                            pass

                    threading.Thread(target=delete_feedback_message, daemon=True).start()
                else:
                    bot.answer_callback_query(call.id, "Спасибо за отзыв! Мы постараемся улучшить ответы.")
                    try:
                        bot.edit_message_reply_markup(
                            chat_id=call.message.chat.id,
                            message_id=call.message.message_id,
                            reply_markup=None
                        )
                    except Exception as edit_error:
                        print(f"Could not edit message reply markup: {str(edit_error)}")
                    
                    sent_message = bot.send_message(
                        call.message.chat.id,
                        "🔄 Спасибо за обратную связь! Мы учтем ваш отзыв для улучшения сервиса.",
                        reply_markup=None if TESTING_MODE else get_cancel_keyboard()
                    )
                    
                    def delete_feedback_message():
                        time.sleep(5)
                        try:
                            bot.delete_message(call.message.chat.id, sent_message.message_id)
                        except Exception as e:
                            pass

                    threading.Thread(target=delete_feedback_message, daemon=True).start()
                    
            else:
                bot.answer_callback_query(call.id, "Ошибка обработки отзыва")
                
        except Exception as e:
            print(f"Error handling feedback: {e}")
            bot.answer_callback_query(call.id, "Произошла ошибка при обработке вашего отзыва")
    
    # Handle callback to respond to support messages
    elif call.data.startswith('reply_'):
        try:
            ticket_id = call.data.split('_')[1]
            
            bot.answer_callback_query(call.id, "Переход к тикету...")
            
            chat_id = call.message.chat.id
            user_states[chat_id] = "support"
            
            # Check if this is an active ticket
            has_ticket = False
            if chat_id in user_support_chats and str(user_support_chats[chat_id].get("topic_message_id")) == ticket_id:
                has_ticket = True
            
            if has_ticket:
                keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
                buttons = [
                    KeyboardButton("❌ Закрыть тикет"),
                    KeyboardButton("👋 Выйти из поддержки")
                ]
                keyboard.add(*buttons)
                
                bot.send_message(
                    chat_id,
                    f"🎫 <b>Вы перешли к тикету #{ticket_id}</b>\n\n"
                    f"Вы можете продолжать общение с технической поддержкой. Просто напишите ваше сообщение.",
                    parse_mode="HTML",
                    reply_markup=keyboard
                )
            else:
                bot.send_message(
                    chat_id,
                    "Не удалось найти указанный тикет. Возможно, он был закрыт.",
                    reply_markup=get_main_menu_keyboard()
                )
                user_states[chat_id] = "normal"
                
                save_support_data()
        except Exception as e:
            bot.answer_callback_query(call.id, "Произошла ошибка")
    else:
        bot.answer_callback_query(call.id)

# Save query logs
def save_query_log(log_data):
    try:
        logs = []
        if os.path.exists(LOGS_FILE):
            try:
                with open(LOGS_FILE, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        logs = loaded_data
                    else:
                        print(f"Warning: {LOGS_FILE} contains invalid data format. Creating new log array.")
                        logs = []
            except json.JSONDecodeError:
                print(f"Warning: {LOGS_FILE} contains invalid JSON. Creating new log array.")
                logs = []
        
        logs.append(log_data)
        
        with open(LOGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        print(f"Query log saved to {LOGS_FILE}")
    except Exception as e:
        print(f"Error saving query log: {str(e)}")
        try:
            with open(LOGS_FILE, 'w', encoding='utf-8') as f:
                json.dump([log_data], f, ensure_ascii=False, indent=2)
            print(f"Recreated {LOGS_FILE} with new log entry")
        except Exception as e2:
            print(f"Failed to recreate log file: {str(e2)}")

# Admin command to toggle testing mode
@bot.message_handler(commands=['testing_mode'])
def toggle_testing_mode(message):
    if str(message.chat.id) != str(ADMIN_CHAT_ID):
        bot.reply_to(message, "❌ Эта команда доступна только администратору.")
        return
    
    parts = message.text.split()
    if len(parts) > 1 and parts[1].lower() in ['on', 'off']:
        param = parts[1].lower()
        if param == 'on':
            enable_testing_mode()
            setup_bot_commands()  # Re-setup commands to apply changes
            bot.reply_to(message, "✅ Режим тестирования ВКЛЮЧЕН. Все команды и интерфейс отключены.")
        else:  # off
            disable_testing_mode()
            bot.reply_to(message, "✅ Режим тестирования ВЫКЛЮЧЕН. Обычный режим работы восстановлен.")
    else:
        # Toggle current mode
        if TESTING_MODE:
            disable_testing_mode()
            bot.reply_to(message, "✅ Режим тестирования ВЫКЛЮЧЕН. Обычный режим работы восстановлен.")
        else:
            enable_testing_mode()
            bot.reply_to(message, "✅ Режим тестирования ВКЛЮЧЕН. Все команды и интерфейс отключены.")

bot.infinity_polling()