from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import time
import shutil
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch

# Initialize Flask app
app = Flask(__name__)
app.config.update({
    'SECRET_KEY': 'dev-key-12345',
    'UPLOAD_FOLDER': 'uploads',
    'TRAINING_DATA_FOLDER': 'training_data',
    'KNOWLEDGE_BASES_FOLDER': 'knowledge_bases',
    'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
    'BASE_LLM_MODEL': 'facebook/opt-350m',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB max upload
})

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAINING_DATA_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNOWLEDGE_BASES_FOLDER'], exist_ok=True)

class RAGApp:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=app.config['EMBEDDING_MODEL'])
        self.vector_stores = {}  # Store vector stores for different knowledge bases
        self.qa_chains = {}      # Store QA chains for different knowledge bases
        self.base_model = None
        self.tokenizer = None
        self.fine_tuned_models = {}  # Store fine-tuned models
        
    def load_documents(self, file_paths):
        documents = []
        for file_path in file_paths:
            try:
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    continue
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return documents
    
    def process_knowledge_base(self, kb_name, documents, chunk_size=1000, chunk_overlap=200):
        """Process documents for a specific knowledge base"""
        kb_path = os.path.join(app.config['KNOWLEDGE_BASES_FOLDER'], kb_name)
        os.makedirs(kb_path, exist_ok=True)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(os.path.join(kb_path, 'index'))
        
        self.vector_stores[kb_name] = vector_store
        return len(chunks)
    
    def load_knowledge_base(self, kb_name):
        """Load a specific knowledge base"""
        kb_path = os.path.join(app.config['KNOWLEDGE_BASES_FOLDER'], kb_name, 'index')
        if os.path.exists(kb_path):
            vector_store = FAISS.load_local(kb_path, self.embeddings)
            self.vector_stores[kb_name] = vector_store
            return True
        return False
    
    def setup_qa_chain(self, kb_name, use_fine_tuned=False, model_name=None):
        """Setup QA chain for a specific knowledge base"""
        if kb_name not in self.vector_stores:
            return False
        
        template = """Use the following context to answer the question:
        Context: {context}
        Question: {question}
        Answer:"""
        
        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        
        if use_fine_tuned and model_name and model_name in self.fine_tuned_models:
            llm = HuggingFacePipeline(pipeline=self.create_pipeline(model_name))
        else:
            llm = HuggingFacePipeline.from_model_id(
                model_id=app.config['BASE_LLM_MODEL'],
                task="text-generation",
                model_kwargs={"temperature": 0.1, "max_length": 512, "do_sample": False}
            )
        
        self.qa_chains[kb_name] = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vector_stores[kb_name].as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        return True
    
    def query_knowledge_base(self, kb_name, query):
        """Query a specific knowledge base"""
        if kb_name not in self.qa_chains:
            return {'error': 'QA chain not initialized for this knowledge base'}
        
        try:
            result = self.qa_chains[kb_name].invoke({"query": query})
            return {
                'answer': result['result'],
                'sources': [{'content': doc.page_content[:200] + '...', 'metadata': doc.metadata} 
                           for doc in result['source_documents']]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def load_base_model(self):
        """Load base model for fine-tuning"""
        if self.base_model is None:
            model_name = app.config['BASE_LLM_MODEL']
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto"
            )
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        return self.base_model
    
    def prepare_fine_tuning_data(self, training_data):
        """Prepare dataset for fine-tuning"""
        formatted_data = []
        for item in training_
            input_text = item.get('input', '')
            context = item.get('context', '')
            output = item.get('output', '')
            
            instruction = f"Context: {context}\nQuestion: {input_text}" if context else f"Question: {input_text}"
            formatted_data.append({'text': f"{instruction}\nAnswer: {output}"})
        
        return Dataset.from_dict({'text': [item['text'] for item in formatted_data]})
    
    def fine_tune_model(self, training_data, model_name="fine_tuned_model", epochs=1):
        """Fine-tune model with given training data"""
        dataset = self.prepare_fine_tuning_data(training_data)
        model_path = os.path.join(app.config['TRAINING_DATA_FOLDER'], model_name)
        os.makedirs(model_path, exist_ok=True)
        
        if self.base_model is None:
            self.load_base_model()
        
        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(self.base_model, lora_config)
        
        training_args = TrainingArguments(
            output_dir=model_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to="none"
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        self.fine_tuned_models[model_name] = model
        return model_path
    
    def create_pipeline(self, model_name):
        """Create pipeline for fine-tuned model"""
        if model_name not in self.fine_tuned_models:
            return None
        
        return pipeline(
            "text-generation",
            model=self.fine_tuned_models[model_name],
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.1,
            do_sample=False
        )
    
    def get_available_knowledge_bases(self):
        """Get list of available knowledge bases"""
        kb_path = app.config['KNOWLEDGE_BASES_FOLDER']
        return [d for d in os.listdir(kb_path) 
                if os.path.isdir(os.path.join(kb_path, d)) and 
                os.path.exists(os.path.join(kb_path, d, 'index'))]
    
    def get_available_models(self):
        """Get list of available fine-tuned models"""
        models_path = app.config['TRAINING_DATA_FOLDER']
        return [d for d in os.listdir(models_path) 
                if os.path.isdir(os.path.join(models_path, d))]

# Initialize the application
rag_app = RAGApp()

@app.route('/')
def index():
    knowledge_bases = rag_app.get_available_knowledge_bases()
    models = rag_app.get_available_models()
    return render_template('index.html', knowledge_bases=knowledge_bases, models=models)

@app.route('/knowledge-base/<kb_name>/upload', methods=['POST'])
def upload_to_knowledge_base(kb_name):
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    uploaded_paths = []
    
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        uploaded_paths.append(file_path)
    
    if not uploaded_paths:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    try:
        documents = rag_app.load_documents(uploaded_paths)
        chunk_count = rag_app.process_knowledge_base(kb_name, documents)
        
        # Load the knowledge base and setup QA chain
        rag_app.load_knowledge_base(kb_name)
        rag_app.setup_qa_chain(kb_name)
        
        # Clean up uploaded files
        for file_path in uploaded_paths:
            os.remove(file_path)
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(documents)} documents into {chunk_count} chunks for knowledge base "{kb_name}"',
            'chunks': chunk_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge-base/<kb_name>/load', methods=['POST'])
def load_knowledge_base(kb_name):
    try:
        success = rag_app.load_knowledge_base(kb_name)
        if success:
            rag_app.setup_qa_chain(kb_name)
            return jsonify({'success': True, 'message': f'Knowledge base "{kb_name}" loaded successfully'})
        else:
            return jsonify({'error': f'Knowledge base "{kb_name}" not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge-base/<kb_name>/query', methods=['POST'])
def query_knowledge_base(kb_name):
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        result = rag_app.query_knowledge_base(kb_name, query)
        if 'error' in result:
            return jsonify(result), 500
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge-base/create', methods=['POST'])
def create_knowledge_base():
    data = request.get_json()
    kb_name = data.get('kb_name', '').strip()
    
    if not kb_name:
        return jsonify({'error': 'Knowledge base name is required'}), 400
    
    kb_path = os.path.join(app.config['KNOWLEDGE_BASES_FOLDER'], kb_name)
    if os.path.exists(kb_path):
        return jsonify({'error': 'Knowledge base already exists'}), 400
    
    os.makedirs(kb_path, exist_ok=True)
    return jsonify({'success': True, 'message': f'Knowledge base "{kb_name}" created successfully'})

@app.route('/training-data/upload', methods=['POST'])
def upload_training_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.json'):
        return jsonify({'error': 'Only JSON files are supported'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Read and validate the training data
        with open(file_path, 'r') as f:
            training_data = json.load(f)
        
        # Save to training data folder
        training_data_name = os.path.splitext(filename)[0]
        training_data_path = os.path.join(app.config['TRAINING_DATA_FOLDER'], training_data_name)
        os.makedirs(training_data_path, exist_ok=True)
        
        with open(os.path.join(training_data_path, 'training_data.json'), 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'message': f'Training data "{training_data_name}" uploaded successfully',
            'sample_count': len(training_data) if isinstance(training_data, list) else 0,
            'training_data_name': training_data_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training-data/<data_name>/finetune', methods=['POST'])
def finetune_with_data(data_name):
    data = request.get_json()
    model_name = data.get('model_name', '').strip()
    epochs = data.get('epochs', 1)
    
    if not model_name:
        return jsonify({'error': 'Model name is required'}), 400
    
    try:
        # Load training data
        training_data_path = os.path.join(app.config['TRAINING_DATA_FOLDER'], data_name, 'training_data.json')
        if not os.path.exists(training_data_path):
            return jsonify({'error': 'Training data not found'}), 404
        
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        # Fine-tune the model
        model_path = rag_app.fine_tune_model(training_data, model_name, epochs)
        
        return jsonify({
            'success': True,
            'message': f'Model "{model_name}" fine-tuned successfully',
            'model_path': model_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training-data/sample', methods=['GET'])
def sample_training_data():
    sample_data = [
        {
            "input": "What is RAG?",
            "context": "RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation.",
            "output": "RAG (Retrieval-Augmented Generation) is a technique that enhances language models by retrieving relevant information from external knowledge sources before generating responses."
        },
        {
            "input": "Why use caching?",
            "context": "Caching stores frequently accessed data to improve performance.",
            "output": "Caching improves application performance by storing frequently accessed data in memory, reducing the need for expensive computations or database queries."
        },
        {
            "input": "What are the benefits of modular design?",
            "context": "Modular design separates functionality into independent, interchangeable components.",
            "output": "Modular design offers benefits like easier maintenance, better code reuse, improved scalability, and simplified testing and debugging."
        }
    ]
    return jsonify({'success': True, 'sample_data': sample_data})

@app.route('/system-info', methods=['GET'])
def system_info():
    return jsonify({
        'success': True,
        'knowledge_bases': rag_app.get_available_knowledge_bases(),
        'models': rag_app.get_available_models(),
        'training_datasets': [d for d in os.listdir(app.config['TRAINING_DATA_FOLDER']) 
                             if os.path.isdir(os.path.join(app.config['TRAINING_DATA_FOLDER'], d))]
    })

if __name__ == '__main__':
    # Create templates directory and basic template
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced RAG System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .kb-item { cursor: pointer; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
        .kb-item:hover { background-color: #f8f9fa; }
        .active-kb { background-color: #e3f2fd; border-color: #2196f3; }
        .model-item { cursor: pointer; padding: 8px; margin: 3px 0; border: 1px solid #ddd; border-radius: 4px; }
        .model-item:hover { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Advanced RAG System</h1>
        
        <div class="row">
            <div class="col-md-4">
                <div class="section">
                    <h3>Knowledge Bases</h3>
                    <div class="mb-3">
                        <input type="text" id="newKbName" class="form-control" placeholder="New knowledge base name">
                        <button class="btn btn-primary mt-2" onclick="createKnowledgeBase()">Create Knowledge Base</button>
                    </div>
                    <div id="kbList">
                        {% for kb in knowledge_bases %}
                        <div class="kb-item" onclick="selectKnowledgeBase('{{ kb }}')">
                            {{ kb }}
                        </div>
                        {% endfor %}
                    </div>
                    <div id="kbUploadSection" class="mt-3" style="display: none;">
                        <input type="file" id="kbFiles" class="form-control" multiple>
                        <button class="btn btn-success mt-2" onclick="uploadToKnowledgeBase()">Upload Documents</button>
                        <div id="kbUploadResult"></div>
                    </div>
                    <button class="btn btn-info mt-2" onclick="loadSelectedKnowledgeBase()">Load Selected KB</button>
                </div>
                
                <div class="section">
                    <h3>Training Data</h3>
                    <input type="file" id="trainingFile" class="form-control" accept=".json">
                    <button class="btn btn-primary mt-2" onclick="uploadTrainingData()">Upload Training Data</button>
                    <button class="btn btn-secondary mt-2" onclick="getSampleTrainingData()">Get Sample Data</button>
                    <div id="trainingUploadResult"></div>
                </div>
                
                <div class="section">
                    <h3>Fine-tuned Models</h3>
                    <div id="modelList">
                        {% for model in models %}
                        <div class="model-item">
                            {{ model }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="section">
                    <h3>Query Selected Knowledge Base</h3>
                    <div class="mb-3">
                        <textarea id="queryInput" class="form-control" rows="3" placeholder="Ask a question..."></textarea>
                    </div>
                    <div class="mb-3">
                        <select id="modelSelect" class="form-control">
                            <option value="base">Base Model</option>
                            {% for model in models %}
                            <option value="{{ model }}">{{ model }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <button class="btn btn-primary" onclick="queryKnowledgeBase()">Ask Question</button>
                    <div id="queryResult" class="mt-3"></div>
                </div>
                
                <div class="section">
                    <h3>Fine-tuning</h3>
                    <div class="mb-3">
                        <select id="trainingDataSelect" class="form-control">
                            <option value="">Select Training Data</option>
                            {% for dataset in [] %}{% endfor %}
                        </select>
                    </div>
                    <div class="mb-3">
                        <input type="text" id="finetuneModelName" class="form-control" placeholder="Model name for fine-tuning">
                        <input type="number" id="finetuneEpochs" class="form-control mt-2" placeholder="Number of epochs" value="1">
                    </div>
                    <button class="btn btn-success" onclick="startFineTuning()">Start Fine-tuning</button>
                    <div id="finetuneResult"></div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let selectedKb = '';
        let selectedTrainingData = '';
        
        function createKnowledgeBase() {
            const kbName = document.getElementById('newKbName').value.trim();
            if (!kbName) {
                alert('Please enter a knowledge base name');
                return;
            }
            
            axios.post('/knowledge-base/create', { kb_name: kbName })
                .then(response => {
                    alert(response.data.message);
                    location.reload(); // Refresh to show new KB
                })
                .catch(error => {
                    alert('Error: ' + (error.response?.data?.error || error.message));
                });
        }
        
        function selectKnowledgeBase(kbName) {
            selectedKb = kbName;
            document.getElementById('kbList').innerHTML = document.getElementById('kbList').innerHTML
                .replace(/active-kb/g, '')
                .replace(kbName, `<span class="active-kb">${kbName}</span>`);
            document.getElementById('kbUploadSection').style.display = 'block';
        }
        
        function uploadToKnowledgeBase() {
            if (!selectedKb) {
                alert('Please select a knowledge base first');
                return;
            }
            
            const files = document.getElementById('kbFiles').files;
            if (files.length === 0) {
                alert('Please select files to upload');
                return;
            }
            
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }
            
            axios.post(`/knowledge-base/${selectedKb}/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })
            .then(response => {
                document.getElementById('kbUploadResult').innerHTML = 
                    `<div class="alert alert-success">${response.data.message}</div>`;
            })
            .catch(error => {
                document.getElementById('kbUploadResult').innerHTML = 
                    `<div class="alert alert-danger">Error: ${error.response?.data?.error || error.message}</div>`;
            });
        }
        
        function loadSelectedKnowledgeBase() {
            if (!selectedKb) {
                alert('Please select a knowledge base first');
                return;
            }
            
            axios.post(`/knowledge-base/${selectedKb}/load`)
                .then(response => {
                    alert(response.data.message);
                })
                .catch(error => {
                    alert('Error: ' + (error.response?.data?.error || error.message));
                });
        }
        
        function queryKnowledgeBase() {
            if (!selectedKb) {
                alert('Please select and load a knowledge base first');
                return;
            }
            
            const query = document.getElementById('queryInput').value.trim();
            const model = document.getElementById('modelSelect').value;
            
            if (!query) {
                alert('Please enter a query');
                return;
            }
            
            axios.post(`/knowledge-base/${selectedKb}/query`, {
                query: query
            })
            .then(response => {
                let resultHtml = `<div class="card"><div class="card-body"><h5>Answer:</h5><p>${response.data.answer}</p>`;
                
                if (response.data.sources && response.data.sources.length > 0) {
                    resultHtml += '<h5 class="mt-3">Sources:</h5><ul>';
                    response.data.sources.forEach(source => {
                        resultHtml += `<li><strong>Content:</strong> ${source.content}<br><strong>Meta</strong> ${JSON.stringify(source.metadata)}</li>`;
                    });
                    resultHtml += '</ul>';
                }
                resultHtml += '</div></div>';
                
                document.getElementById('queryResult').innerHTML = resultHtml;
            })
            .catch(error => {
                document.getElementById('queryResult').innerHTML = 
                    `<div class="alert alert-danger">Error: ${error.response?.data?.error || error.message}</div>`;
            });
        }
        
        function uploadTrainingData() {
            const file = document.getElementById('trainingFile').files[0];
            if (!file) {
                alert('Please select a training data file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            axios.post('/training-data/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })
            .then(response => {
                document.getElementById('trainingUploadResult').innerHTML = 
                    `<div class="alert alert-success">${response.data.message}</div>`;
                selectedTrainingData = response.data.training_data_name;
                updateTrainingDataSelect();
            })
            .catch(error => {
                document.getElementById('trainingUploadResult').innerHTML = 
                    `<div class="alert alert-danger">Error: ${error.response?.data?.error || error.message}</div>`;
            });
        }
        
        function getSampleTrainingData() {
            axios.get('/training-data/sample')
                .then(response => {
                    const dataStr = JSON.stringify(response.data.sample_data, null, 2);
                    const blob = new Blob([dataStr], {type: 'application/json'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'sample_training_data.json';
                    a.click();
                    URL.revokeObjectURL(url);
                })
                .catch(error => {
                    alert('Error: ' + (error.response?.data?.error || error.message));
                });
        }
        
        function startFineTuning() {
            const trainingDataName = selectedTrainingData;
            const modelName = document.getElementById('finetuneModelName').value.trim();
            const epochs = parseInt(document.getElementById('finetuneEpochs').value) || 1;
            
            if (!trainingDataName) {
                alert('Please upload training data first');
                return;
            }
            
            if (!modelName) {
                alert('Please enter a model name');
                return;
            }
            
            axios.post(`/training-data/${trainingDataName}/finetune`, {
                model_name: modelName,
                epochs: epochs
            })
            .then(response => {
                document.getElementById('finetuneResult').innerHTML = 
                    `<div class="alert alert-success">${response.data.message}</div>`;
                location.reload(); // Refresh to show new model
            })
            .catch(error => {
                document.getElementById('finetuneResult').innerHTML = 
                    `<div class="alert alert-danger">Error: ${error.response?.data?.error || error.message}</div>`;
            });
        }
        
        function updateTrainingDataSelect() {
            axios.get('/system-info')
                .then(response => {
                    const select = document.getElementById('trainingDataSelect');
                    select.innerHTML = '<option value="">Select Training Data</option>';
                    response.data.training_datasets.forEach(dataset => {
                        const option = document.createElement('option');
                        option.value = dataset;
                        option.textContent = dataset;
                        select.appendChild(option);
                    });
                });
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            updateTrainingDataSelect();
        });
    </script>
</body>
</html>
        ''')
    
    print("Starting application...")
    print(f"Training data folder: {app.config['TRAINING_DATA_FOLDER']}")
    print(f"Knowledge bases folder: {app.config['KNOWLEDGE_BASES_FOLDER']}")
    print("Visit http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)