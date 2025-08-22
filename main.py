# Carregar variáveis do arquivo .env PRIMEIRO
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
from typing import List, Dict, Any, Optional
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec
import arxiv
from datetime import datetime
import logging
import re

# Tentar importar nest_asyncio para loops aninhados
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("✅ nest_asyncio configurado")
except ImportError:
    print("⚠️ nest_asyncio não disponível - instale com: pip install nest-asyncio")

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustEmbeddingManager:
    """Gerenciador robusto de embeddings com fallbacks."""
    
    @staticmethod
    def get_embedding_model():
        """Tenta diferentes modelos de embedding em ordem de preferência."""
        
        embedding_options = [
            {
                "name": "intfloat/multilingual-e5-large",
                "description": "E5 Large Multilingual - Melhor qualidade"
            },
            {
                "name": "intfloat/multilingual-e5-base", 
                "description": "E5 Base Multilingual - Balanceado"
            },
            {
                "name": "intfloat/multilingual-e5-small",
                "description": "E5 Small Multilingual - Mais rápido"
            },
            {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "MiniLM Multilingual - Compatibilidade"
            },
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "All-MiniLM - Fallback confiável"
            }
        ]
        
        for option in embedding_options:
            try:
                logger.info(f"Tentando carregar: {option['name']} - {option['description']}")
                
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                embed_model = HuggingFaceEmbedding(
                    model_name=option["name"],
                    max_length=512,
                    normalize=True
                )
                
                logger.info(f"✅ Sucesso! Usando: {option['name']}")
                return embed_model
                
            except Exception as e:
                logger.warning(f"❌ Falhou {option['name']}: {str(e)[:100]}")
                continue
        
        logger.info("🔄 Usando embedding padrão do LlamaIndex")
        return None

class ImprovedFinancialAgent:
    def __init__(self):
        print("🚀 Inicializando Agente IA Melhorado...")
        
        # Verificar se as chaves estão carregadas
        groq_key = os.getenv("GROQ_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        print(f"🔑 GROQ_API_KEY: {'✅ Carregada' if groq_key else '❌ Não encontrada'}")
        print(f"🔑 TAVILY_API_KEY: {'✅ Carregada' if tavily_key else '⚠️ Não encontrada (opcional)'}")
        
        if not groq_key:
            raise ValueError("GROQ_API_KEY não encontrada no arquivo .env")
        
        # Configuração do LLM
        self.llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0.1
        )
        print("✅ LLM Groq configurado")
        
        # Configuração robusta de embeddings
        embed_model = RobustEmbeddingManager.get_embedding_model()
        if embed_model:
            Settings.embed_model = embed_model
        
        Settings.llm = self.llm
        
        # Inicializar ferramentas
        self.setup_tools()
        print(f"✅ {len(self.tools)} ferramentas configuradas")
        
        # Criar agente ReAct
        self.agent = self.create_react_agent()
        print("✅ Agente ReAct inicializado")

    def imposto_renda_melhorado(self, rendimento: float, ano: int = 2024) -> str:
        """Calcula o imposto de renda com validações robustas."""
        try:
            if rendimento < 0:
                return "❌ Rendimento não pode ser negativo."
            
            # Tabela 2024 (valores reais)
            if ano == 2024:
                faixas = [
                    (28559.70, 0.0, 0, "Isento"),
                    (42253.25, 0.075, 2141.98, "7,5%"),
                    (56717.56, 0.15, 5304.90, "15%"),
                    (74414.84, 0.225, 9756.12, "22,5%"),
                    (float('inf'), 0.275, 14067.51, "27,5%")
                ]
            else:
                # Valores simplificados para outros anos
                faixas = [
                    (28000, 0.0, 0, "Isento"),
                    (42000, 0.075, 2100, "7,5%"),
                    (56000, 0.15, 5300, "15%"),
                    (74000, 0.225, 9700, "22,5%"),
                    (float('inf'), 0.275, 14000, "27,5%")
                ]
            
            for limite, aliquota, deducao, faixa_nome in faixas:
                if rendimento <= limite:
                    imposto_bruto = rendimento * aliquota
                    imposto_devido = max(0, imposto_bruto - deducao)
                    
                    resultado = f"""
🧮 **CÁLCULO DO IMPOSTO DE RENDA {ano}**
{'='*45}
💰 **Rendimento Bruto**: R$ {rendimento:,.2f}
📊 **Faixa**: {faixa_nome}
📈 **Alíquota**: {aliquota*100:.1f}%
➖ **Dedução**: R$ {deducao:,.2f}
💸 **Imposto Devido**: R$ {imposto_devido:,.2f}
💵 **Renda Líquida**: R$ {rendimento - imposto_devido:,.2f}
{'='*45}
                    """
                    
                    if imposto_devido == 0:
                        resultado += "\n✅ **ISENTO DE IMPOSTO DE RENDA!**"
                    else:
                        percentual_efetivo = (imposto_devido / rendimento) * 100
                        resultado += f"\n📊 **Alíquota Efetiva**: {percentual_efetivo:.2f}%"
                    
                    return resultado.strip()
                    
        except Exception as e:
            return f"❌ Erro no cálculo: {str(e)}"

    def calculadora_financeira_geral(self, tipo: str, **kwargs) -> str:
        """Calculadora financeira para diversos cálculos."""
        try:
            if tipo.lower() == "porcentagem":
                valor = kwargs.get('valor', 0)
                percentual = kwargs.get('percentual', 0)
                
                if valor <= 0 or percentual <= 0:
                    return "❌ Valores devem ser positivos para cálculo de porcentagem"
                
                resultado = valor * (percentual / 100)
                
                return f"""
📊 **CÁLCULO DE PORCENTAGEM**
{'='*30}
💰 **Valor Base**: {valor:,.2f}
📈 **Percentual**: {percentual}%
💵 **Resultado**: {resultado:,.2f}
📋 **Fórmula**: {valor:,.2f} × {percentual}% = {resultado:,.2f}
                """.strip()
                
            elif tipo.lower() == "juros_compostos":
                principal = kwargs.get('principal', 0)
                taxa = kwargs.get('taxa', 0) / 100  # converter percentual
                periodo = kwargs.get('periodo', 0)
                
                if principal <= 0 or taxa <= 0 or periodo <= 0:
                    return "❌ Valores devem ser positivos para juros compostos"
                
                montante = principal * (1 + taxa) ** periodo
                juros = montante - principal
                
                return f"""
📈 **JUROS COMPOSTOS**
{'='*30}
💰 **Capital Inicial**: R$ {principal:,.2f}
📊 **Taxa**: {taxa*100:.2f}% ao período
⏱️ **Período**: {periodo} períodos
💸 **Montante Final**: R$ {montante:,.2f}
💵 **Juros Ganhos**: R$ {juros:,.2f}
📈 **Rendimento**: {((montante/principal-1)*100):.2f}%
                """.strip()
                
            elif tipo.lower() == "juros_simples":
                principal = kwargs.get('principal', 0)
                taxa = kwargs.get('taxa', 0) / 100
                periodo = kwargs.get('periodo', 0)
                
                juros = principal * taxa * periodo
                montante = principal + juros
                
                return f"""
📊 **JUROS SIMPLES**
{'='*25}
💰 **Capital**: R$ {principal:,.2f}
📊 **Taxa**: {taxa*100:.2f}%
⏱️ **Período**: {periodo}
💸 **Juros**: R$ {juros:,.2f}
💵 **Montante**: R$ {montante:,.2f}
                """.strip()
                
            else:
                return f"❌ Tipo '{tipo}' não suportado. Use: juros_compostos, juros_simples"
                
        except Exception as e:
            return f"❌ Erro no cálculo financeiro: {str(e)}"

    def consulta_arxiv_melhorada(self, query: str, max_results: int = 3) -> str:
        """Consulta robusta ao arXiv."""
        try:
            search = arxiv.Search(
                query=query.strip(),
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            resultados = []
            for i, artigo in enumerate(search.results(), 1):
                resumo = artigo.summary.replace('\n', ' ').strip()
                if len(resumo) > 250:
                    resumo = resumo[:250] + "..."
                
                autores = [autor.name for autor in artigo.authors[:3]]
                if len(artigo.authors) > 3:
                    autores.append("et al.")
                
                resultado = f"""
📄 **ARTIGO {i}**
📝 **Título**: {artigo.title.strip()}
👥 **Autores**: {', '.join(autores)}
📂 **Categoria**: {artigo.primary_category}
📅 **Data**: {artigo.published.strftime('%d/%m/%Y')}
🔗 **Link**: {artigo.entry_id}
📖 **Resumo**: {resumo}
                """.strip()
                resultados.append(resultado)
            
            if resultados:
                header = f"🔍 **RESULTADOS PARA: '{query}'**\n{'='*60}"
                return header + "\n\n" + "\n\n".join(resultados)
            else:
                return f"❌ Nenhum artigo encontrado para: '{query}'"
                
        except Exception as e:
            return f"❌ Erro na consulta ao arXiv: {str(e)}"

    def busca_web_inteligente(self, query: str, max_results: int = 3) -> str:
        """Busca web com tratamento robusto de erros."""
        try:
            tavily_key = os.environ.get("TAVILY_API_KEY")
            if not tavily_key:
                return """
❌ **CHAVE TAVILY NÃO CONFIGURADA**

Para usar busca web:
1. Adicione TAVILY_API_KEY="sua_chave" no arquivo .env
2. Obtenha uma chave em: https://tavily.com
                """.strip()
            
            tavily_tool = TavilyToolSpec(api_key=tavily_key)
            resultados_raw = tavily_tool.search(query, max_results=max_results)
            
            return f"🌐 **Busca Web para**: {query}\n\n{str(resultados_raw)}"
            
        except Exception as e:
            return f"❌ Erro na busca web: {str(e)}"

    def setup_tools(self):
        """Configurar ferramentas do agente."""
        self.tools = [
            FunctionTool.from_defaults(
                fn=self.imposto_renda_melhorado,
                name="calcular_imposto_renda",
                description=(
                    "🧮 Calcula imposto de renda brasileiro com tabela oficial 2024. "
                    "Parâmetros: rendimento (float), ano (int, opcional). "
                    "Retorna cálculo detalhado com todas as faixas e deduções."
                )
            ),
            FunctionTool.from_defaults(
                fn=self.calculadora_financeira_geral,
                name="calculadora_financeira",
                description=(
                    "📊 Calculadora financeira geral para diversos cálculos. "
                    "Parâmetros: tipo (str), principal/valor (float), taxa/percentual (float), periodo (int). "
                    "Tipos: 'porcentagem', 'juros_compostos', 'juros_simples'."
                )
            ),
            FunctionTool.from_defaults(
                fn=self.consulta_arxiv_melhorada,
                name="consultar_artigos_cientificos",
                description=(
                    "📚 Busca artigos científicos no arXiv por relevância. "
                    "Parâmetros: query (str), max_results (int, opcional). "
                    "Retorna artigos acadêmicos com resumo, autores e links."
                )
            ),
            FunctionTool.from_defaults(
                fn=self.busca_web_inteligente,
                name="buscar_informacoes_web",
                description=(
                    "🌐 Busca informações atuais na web usando Tavily. "
                    "Parâmetros: query (str), max_results (int, opcional). "
                    "Retorna informações web atualizadas e relevantes."
                )
            )
        ]

    def create_react_agent(self) -> ReActAgent:
        """Criar agente ReAct otimizado."""
        
        system_prompt = """
🤖 **ASSISTENTE IA MULTIFUNCIONAL**

Você é um assistente inteligente especializado em:

📊 **FINANÇAS E ECONOMIA**
- Cálculos de Imposto de Renda brasileiro
- Juros compostos e simples
- Análises de investimentos
- Planejamento financeiro

📚 **PESQUISA ACADÊMICA**
- Busca de artigos científicos no arXiv
- Análise de papers e publicações
- Resumos de pesquisas

🌐 **INFORMAÇÕES ATUAIS**
- Busca web para dados recentes
- Notícias e tendências
- Informações de mercado

🧠 **CONHECIMENTO GERAL**
- Matemática e estatística
- Tecnologia e programação
- Ciências em geral
- Explicações didáticas

**INSTRUÇÕES:**
1. SEMPRE use as ferramentas disponíveis quando apropriado
2. Para cálculos financeiros, use as funções específicas
3. Para perguntas gerais, responda com seu conhecimento
4. Seja detalhado e didático nas explicações
5. Formate as respostas de forma clara e organizada

**EXAMPLES:**
- "Como calcular juros compostos?" → Use calculadora_financeira
- "Busque artigos sobre IA" → Use consultar_artigos_cientificos
- "Qual o IR de R$ 50.000?" → Use calcular_imposto_renda
- "O que é blockchain?" → Responda diretamente

Sempre priorize a precisão e utilidade das informações!
        """.strip()

        try:
            print("🔄 Criando ReActAgent otimizado...")
            agent = ReActAgent.from_tools(
                tools=self.tools,
                llm=self.llm,
                verbose=True,
                system_prompt=system_prompt
            )
            print("✅ ReActAgent criado com sucesso")
            return agent
            
        except Exception as e:
            print(f"⚠️ Erro com from_tools, tentando construtor básico: {e}")
            try:
                agent = ReActAgent(
                    tools=self.tools,
                    llm=self.llm,
                    verbose=True
                )
                print("✅ ReActAgent criado com construtor básico")
                return agent
            except Exception as e2:
                print(f"❌ Erro crítico: {e2}")
                raise e2

    def extrair_valores_numericos(self, texto: str) -> Dict[str, float]:
        """Extrai valores numéricos de texto para cálculos."""
        valores = {}
        
        # Buscar valores monetários
        valores_monetarios = re.findall(r'R?\$?\s*([\d.,]+)', texto)
        if valores_monetarios:
            valores['principal'] = float(valores_monetarios[0].replace('.', '').replace(',', '.'))
            valores['valor'] = valores['principal']  # alias para porcentagem
        
        # Buscar percentuais
        percentuais = re.findall(r'([\d.,]+)\s*%', texto)
        if percentuais:
            valores['taxa'] = float(percentuais[0].replace(',', '.'))
            valores['percentual'] = valores['taxa']  # alias para porcentagem
        
        # Buscar períodos
        periodos = re.findall(r'(\d+)\s*(?:anos?|meses?|dias?)', texto, re.IGNORECASE)
        if periodos:
            valores['periodo'] = float(periodos[0])
        
        return valores

    def chat(self, message: str) -> str:
        """Chat inteligente com prioridade para o agente ReAct."""
        try:
            print(f"🔍 Processando: '{message}'")
            
            # PRIMEIRA TENTATIVA: Sempre usar o agente ReAct
            try:
                print("🤖 Tentando agente ReAct...")
                
                if hasattr(self.agent, 'query'):
                    response = self.agent.query(message)
                elif hasattr(self.agent, 'chat'):
                    response = self.agent.chat(message)
                else:
                    raise AttributeError("Agente não tem método query ou chat")
                
                result = str(response).strip()
                
                # Verificar se a resposta é útil
                if result and len(result) > 20 and not any(erro in result.lower() for erro in ['error', 'erro', 'failed', 'none']):
                    print("✅ ReAct respondeu com sucesso")
                    return result
                else:
                    print("⚠️ Resposta do ReAct inadequada, tentando fallback")
                    
            except Exception as e:
                print(f"⚠️ ReAct falhou: {str(e)[:100]}")
            
            # FALLBACK: Análise manual e uso de ferramentas específicas
            print("🔄 Usando análise manual...")
            
            message_lower = message.lower()
            
            # Detecção específica para IR
            if any(term in message_lower for term in ['imposto de renda', 'calcular ir', 'ir de']):
                print("💰 Detectado: Imposto de Renda")
                valores = self.extrair_valores_numericos(message)
                if valores.get('principal'):
                    return self.imposto_renda_melhorado(valores['principal'])
                else:
                    return "❌ Por favor, informe o valor da renda para calcular o IR"
            
            # Detecção para cálculos de porcentagem
            elif any(term in message_lower for term in ['% de', 'porcent', 'percentual de']):
                print("📊 Detectado: Cálculo de Porcentagem")
                valores = self.extrair_valores_numericos(message)
                if valores.get('valor') and valores.get('percentual'):
                    return self.calculadora_financeira_geral('porcentagem', **valores)
                else:
                    return """
📊 **Para calcular porcentagem, preciso de:**
💰 Valor base (ex: 10.000)
📈 Percentual (ex: 15%)

**Exemplos:**
• "Quanto é 15% de 10.000?"
• "Calcule 20% de R$ 5.000"
• "Qual é 8,5% de 12.500?"
                    """.strip()
            
            # Detecção para juros compostos
            elif any(term in message_lower for term in ['juros compostos', 'compound interest', 'montante']):
                print("📈 Detectado: Juros Compostos")
                valores = self.extrair_valores_numericos(message)
                if all(k in valores for k in ['principal', 'taxa', 'periodo']):
                    return self.calculadora_financeira_geral('juros_compostos', **valores)
                else:
                    return """
📈 **Para calcular juros compostos, preciso de:**
💰 Capital inicial (ex: R$ 10.000)
📊 Taxa de juros (ex: 10% ao ano)
⏱️ Período (ex: 5 anos)

**Exemplo:** "Calcule juros compostos de R$ 10.000 a 10% por 5 anos"
                    """.strip()
            
            # Detecção para artigos científicos
            elif any(term in message_lower for term in ['artigos científicos', 'papers', 'arxiv', 'pesquisa acadêmica']):
                print("📚 Detectado: Busca Acadêmica")
                # Limpar termos da busca
                query = re.sub(r'\b(busque|procure|artigos|científicos|papers|sobre)\b', '', message, flags=re.IGNORECASE).strip()
                return self.consulta_arxiv_melhorada(query)
            
            # Para perguntas incompletas sobre porcentagem
            elif 'porcentagem' in message_lower and any(num in message for num in re.findall(r'\d+', message)):
                print("❓ Detectado: Pergunta incompleta sobre porcentagem")
                numeros = re.findall(r'[\d.,]+', message)
                if numeros:
                    valor = numeros[0].replace('.', '').replace(',', '.')
                    return f"""
❓ **Pergunta incompleta detectada!**

Você mencionou o valor **{valor}**, mas não especificou:
📊 **Qual percentual?** 

**Exemplos corretos:**
• "Quanto é 10% de {valor}?"
• "Calcule 15% de {valor}"
• "Qual é 25% de {valor}?"

💡 **Ou talvez queira saber:**
• "{valor} é quantos % de [outro valor]?"
                    """.strip()
            
            # Resposta para perguntas gerais usando conhecimento do LLM
            else:
                print("🧠 Respondendo com conhecimento geral")
                
                # Usar o LLM diretamente para perguntas gerais
                try:
                    response = self.llm.complete(f"""
Você é um assistente IA especializado. Responda à pergunta de forma clara e didática:

PERGUNTA: {message}

Forneça uma resposta detalhada e bem formatada com emojis quando apropriado.
                    """)
                    
                    return str(response).strip()
                    
                except Exception as e:
                    return f"""
❌ **Não consegui processar sua pergunta adequadamente.**

🤖 **O que posso fazer por você:**
• 💰 Cálculos de Imposto de Renda
• 📈 Cálculos financeiros (juros, investimentos)
• 📚 Busca de artigos científicos
• 🧠 Responder perguntas gerais
• 🌐 Buscar informações atuais (se configurado)

**Erro:** {str(e)[:100]}
                    """.strip()
                
        except Exception as e:
            print(f"❌ Erro crítico no chat: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Erro crítico: {str(e)}"

def main():
    """Função principal melhorada."""
    try:
        print("🔧 Inicializando sistema...")
        
        # Verificar arquivo .env
        if not os.path.exists('.env'):
            print("""
❌ **ARQUIVO .env NÃO ENCONTRADO!**

📝 **Crie um arquivo .env com:**
GROQ_API_KEY="sua_chave_groq_aqui"
TAVILY_API_KEY="sua_chave_tavily_aqui"  # opcional

🔗 **Links:**
• Groq: https://groq.com
• Tavily: https://tavily.com
            """)
            return None
        
        agent = ImprovedFinancialAgent()
        
        print(f"""
🎉 **AGENTE IA INICIALIZADO COM SUCESSO!**

🚀 **CAPACIDADES COMPLETAS:**
💰 Cálculos de Imposto de Renda brasileiro
📈 Cálculos financeiros (juros compostos, simples)
📚 Pesquisa de artigos científicos (arXiv)
🌐 Busca de informações atuais (web)
🧠 Conhecimento geral (matemática, ciências, tecnologia)

💡 **EXEMPLOS DE USO:**
• "Como calcular juros compostos de R$ 10.000 a 12% por 3 anos?"
• "Qual o imposto de renda para R$ 80.000 anuais?"
• "Busque artigos sobre machine learning"
• "O que é blockchain e como funciona?"
• "Explique a teoria da relatividade"
• "Como funciona o mercado de ações?"

🚪 Digite 'sair' para encerrar
        """)
        
        return agent
        
    except Exception as e:
        print(f"\n❌ **ERRO NA INICIALIZAÇÃO**: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    agent = main()
    
    if agent:
        print("\n" + "="*50)
        while True:
            try:
                user_input = input("\n💬 Sua pergunta: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'sair', 'q']:
                    print("\n👋 **Até logo! Obrigado por usar o Agente IA!**")
                    break
                elif not user_input:
                    print("⚠️ Por favor, digite uma pergunta.")
                    continue
                
                print(f"\n🤖 **Processando sua pergunta...**")
                print("-" * 40)
                
                response = agent.chat(user_input)
                
                print(f"\n📋 **Resposta:**")
                print("=" * 50)
                print(response)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 **Sessão encerrada pelo usuário.**")
                break
            except Exception as e:
                print(f"\n❌ **Erro inesperado:** {e}")
                continue