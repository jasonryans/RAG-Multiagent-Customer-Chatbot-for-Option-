## RAG Multiagent Customer Service Chatbot for Option's Billiard & Bistro
Chatbot ini didesain untuk menjawab informasi tentang Option Billiard & Bistro, seperti harga sewa meja billiard, harga menu, alamat tempat, nomor telepon tempat, jam buka dan tutup, serta memberikan tutorial untuk pemula yang ingin bermain billiard. Program ini akan berbasis multiagent dimana akan menggunakan Gemini untuk menangani pertanyaan user tentang harga sewa meja billiard, harga menu, alamat tempat, nomor telepon tempat, serta jam buka dan tutup. Sedangkan, untuk memberikan tutorial untuk pemula yang ingin bermain billiard akan ditangani oleh Cohere.

## Run di VScode
1. Run "python -m venv .venv" di terminal untuk membuat virtual environment
2. Run ".\.venv\Scripts\activate" untuk activate virtual environment di vscode (akan muncul (.venv) di depan terminal)
3. Ctrl+Shft+P → lalu ketik Python: Select Interpreter” → pilih yang akhirannya .venv.
4. Install "pip install streamlit google-generativeai numpy python-dotenv" 
5. Install "pip install cohere"
6. Create file ".env" buat API key
7. Buat API key di Gemini dan Cohere
Link untuk buat akun Gemini menggunakan Google Ai Studio: https://aistudio.google.com/
Link untuk buat akun Cohere: https://cohere.com/  
8. Setelah muncul, masukkan ke dalam .env seperti begini
GEMINI_API_KEY=YOUR_API_KEY
COHERE_API_KEY=YOUR_API_KEY
9. Run "streamlit run app.py" di terminal terlebih dahulu

## Run di GPU3
1. Run "python -m venv .venv" di terminal untuk membuat virtual environment
2. Run ".\.venv\Scripts\activate" untuk activate virtual environment di vscode (akan muncul (.venv) di depan terminal)
3. Install "pip install streamlit google-generativeai numpy python-dotenv" 
4. Install "pip install cohere"
5. Input di terminal "nano.env" buat API key
6. Buat API key di Gemini dan Cohere
Link untuk buat akun Gemini menggunakan Google Ai Studio: https://aistudio.google.com/
Link untuk buat akun Cohere: https://cohere.com/  
7. Setelah itu, masukkan ke dalam .env seperti begini
GEMINI_API_KEY=YOUR_API_KEY
COHERE_API_KEY=YOUR_API_KEY
7. Run "streamlit run app.py" di terminal terlebih dahulu
9. Link setelah run streamlit di terminal GPU3: 
https://gpu3.petra.ac.id/user/user_name/proxy/8501/