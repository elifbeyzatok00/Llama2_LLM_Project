# Explainable AI Project

## Deneyin Amacı
> Hazır modelin, hedeflenen konu (Explainable AI)  hakkındaki sorulan sorulara eğitilirken verilen pdflerdeki verilere dayanarak cevap vermesi. Aynı zamanda verdiği cevabı hangi kaynaklara dayanarak verdiğini göstermesi.

## Kullanılan Teknolojiler
> Modele uygulanan işlemler yüksek RAM ve GPU gerektiren cihazlarda yapılabilir. Bunun için de bir SaaS platformu olan **Google Collab** kullanıldı. Ve bazı satın alımlar gerçekleştirilerek GPU, CPU ve yüksek RAM kullanımı gerçekleştirildi. 
> Kullanılan hazır dil işleme modelinin (**Llama2**) çekilmesi için Hugging Face kütüphanesinin sunduğu bazı API’ler kullanıldı. 
> Hazır dil modelini uzmanlaştırmak için eklenen pdflerdeki verileri chunklara ayırıp hem bu chunkları vector olarak tutması hem de bu chunkların kaynağını (hangi pdf dosyasının kaçıncı sayfasından alındığını) tutması için **Pinecone** veri tabanı kullanıldı.
> Hazır dil modeline sorulan sorulara daha kapsamlı cevap vermesini sağlamak ve sorulan sorulara verdiği cevapları (chunkları) hangi pdf dosyasından aldığını göstermek için **Langchain** kütüphanesi kullanıldı.

### Projenin Anlaşılması İçin Okunması Gerekenler
## LLM modeller
LLM( Large Language Model) yani Büyük Dil Modelleri Nelerdir? Büyük dil modelleri, insan dilini anlamak, oluşturmak ve yönetmek için tasarlanmış yapay zeka destekli sistemlerdir. Bu modeller genellikle onlarca gigabayt boyutundadır ve genellikle derin öğrenme teknikleri kullanılarak oluşturulur; en dikkate değer mimari ise Transformer'dır. Transformer mimarisi, modellerin bir cümledeki kelimelerin bağlamını ve bunların ilişkilerini yakalamasını sağlayarak tutarlı ve bağlamsal olarak alakalı metinler oluşturmalarına olanak tanır.
Büyük dil modelleri kavramı, OpenAI'nin GPT (Generative Pre-trained Transformer) gibi modelleriyle ortaya çıkmaya başladı. Bu modeller şaşırtıcı derecede insana benzeyen metinler üretme yetenekleriyle ün kazandı. Bu büyük dil modelleri; internetten, kitaplardan, makalelerden ve diğer metin kaynaklarından içerik içeren büyük veri kümeleri üzerinde önceden eğitilmiştir. Ön eğitim süreci modellere genel bir dil ve dünya bilgisi anlayışı kazandırır.[1]

 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/c8630ee1-5014-4894-a22b-97748d7a503a)
 
Şekil-1 LLM modeller yapay zekasındaki yerini gösteren görsel  

Büyük Dil Modelleri (LLM'ler) işlevselliği beş alana ayrılabilir: Bilgi Yanıtlama, Çeviri, Metin Oluşturma, Yanıt Oluşturma ve Sınıflandırma.
Günümüzün kurumsal ihtiyaçları açısından tartışmasız en önemli olanı sınıflandırmadır ve metin oluşturma da en etkileyici ve çok yönlü olanıdır.
Çeşitli LLM teklifleri bu beş işlevsellik alanını değişen derecelerde kapsar.
Sınıflandırma, Yanıt Oluşturma, Metin Oluşturma, Çeviri, Bilgi Yanıtlama
Burada bahsedilen teknolojilerin çoğuna HuggingFace sitesi üzerinden erişilebilir .
Bu projede kullanılacak olan LLM Model Llama2’ye de  HuggingFace sitesi üzerinden ulaşılabilir.[2]


 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/dfafc549-9ff8-4e2b-b0cd-97eb4d530d1f)

Şekil-2 Mevcut Büyük Dil Modeli ( LLM ) ortamını gösteren bir grafik 

## Eğitilmiş Dil Modeli Belirleme
Bu proje için ücretsiz ve  (tercihen) Offline(Çevrimdışı) çalışabilen bir modele ihtiyaç vardır.
HuggingFace sitesi birçok alanda birçok hazır model sunan bir site. Buradaki modeller tarafımızca  incelendi ve en uygun, en kullanışlı olan modelin Llama 2 olduğuna karar verildi.
OpenAI’ da hazır modeller sunuyordu fakat bu modeller hem offline olarak çalıştırılamadığından yani lokal cihaza getirilemediğinden hem de ücretli olduklarından dolayı kullanımı tercih edilmedi.
## Llama2 nedir?
Llama 2, Meta şirketinin açık kaynaklı büyük dil modelidir (LLM). Temel olarak, bu, Facebook ana şirketinin OpenAI'nin GPT modellerine ve Google'ın Palm 2 gibi AI modellerine verdiği yanıttır. Ancak önemli bir farkla: neredeyse herkesin araştırma ve ticari amaçlarla kullanması için ücretsiz olarak sunulmaktadır.
Llama 2, GPT-3 ve PaLM 2 gibi LLM ailesindendir. Tüm bu modeller temelde aynı şekilde
geliştirilmiş ve çalışmaktadır. Hepsi aynı transformatör mimarisini ve ön eğitim ve ince ayar gibi geliştirme fikirlerini kullanır.

Bir metin istemine girdiğinizde veya Llama 2'ye başka bir şekilde metin girişi sağladığınızda, kendi sinir ağını kullanarak en makul devam eden metni tahmin etmeye çalışır. Bu, milyarlarca değişken (parametre) içeren basamaklı bir algoritmadır ve insan beyni baz alınarak modellenmiştir. 

Llama 2, tüm farklı parametrelere farklı ağırlıklar atayarak ve biraz rastgelelik ekleyerek inanılmaz derecede insani tepkiler üretebilir. [3]

 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/c64f2302-bac5-4070-91ed-ad33c68ddc1a)
 
Şekil-3 Llama2 logosu 

Llama 2'nin, ChatGPT veya Google Bard gibi gösterişli, kullanımı kolay bir demo uygulaması henüz bulunmuyor . Şimdilik bunu denemenin en iyi yolu , açık kaynaklı yapay zeka modelleri için başvurulacak merkez haline gelen platform olan Hugging Face'tir. Hugging Face aracılığıyla Llama2'nin aşağıdaki sürümlerini deneyebilirsiniz:
●	Llama 2 7B Chat [4]
●	Llama 2 13B Chat [5]
●	Llama 2 70B Chat [6]
Llama2 kurulumu, kullanımı:
Projede Llama2 modelini kullanmak için Hugging Face’in sağladığı API’lerden yararlanıldı. Proje içerisine yazılan aşağıdaki kodlar ile bu işlem gerçekleştirildi:

```py
!pip install langchain huggingface pytorch
import transformers

from huggingface_hub import notebook_login
notebook_login()
```

## Eğitilmiş dil modeli eğitimi / özelleştirilmesi
### LLM Modellerde  Fine Tuning Yapmak (Büyük Dil Modellerinde İnce Ayar Yapmak)
Önceden eğitilmiş LLM modelleri etkileyici dil yeteneklerine sahiptir, ancak belirli görevler veya endüstriler için gereken spesifikliğe gerçekten sahip değildirler. Bu, modellerin ince ayarlanmasıyla başarılabilir. Büyük bir dil modeline ince ayar yapma süreci, genellikle önceden eğitilmiş bir modelin alınmasını ve onu belirli bir göreve, projeye, sektöre, alana veya uygulamaya ilişkin daha odaklanmış bir veri kümesi üzerinde eğitmeyi içerir.

 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/affd0515-8da6-4531-9f00-d66b9c96dfc6)
 
Şekil-4 Llama-2-chat modelinin çalışma şeklini gösteren görsel

Fakat bu noktada Fine Tuning yaparken kullanılan  veri miktarı yeterince büyük değildi. Bu nedenle Fine Tuning işleminden vazgeçildi. Var olan az miktarda veri ile bile modeli belirli bir alanda uzmanlaştırmayı sağlayacak olan Langchain kütüphanesi kullanılmaya karar verildi.
Langchain, doğal dil işleme (NLP) görevleri için açık kaynaklı bir Python kütüphanesidir. Çok çeşitli NLP görevlerini destekler, bunlara metin sınıflandırma, metin özeti, soru cevaplama ve makine çevirisi dahildir. 
Langchain, bir dizi farklı NLP modelini ve algoritmasını kullanır. Bu modeller, metin ve koddan oluşan büyük bir veri kümesi üzerinde eğitilir. Langchain, bu modelleri bir dizi NLP görevi için kullanmak için bir arabirim sağlar.
Langchain'i kullanmak için, bir model seçmeniz ve onu bir metin veya kod parçasıyla eğitmeniz gerekir. Ardından, modelinizi bir NLP görevi gerçekleştirmek için kullanabilirsiniz.[7]
 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/b1bbbf78-aa86-42b8-ac54-70ca84448762)
 
Şekil-5 Langchain framworkünün çalışma şeklini gösteren görsel

Langchain kütüphanesi gönderilen pdfleri chunklara yani küçük metin parçacıklarına ayırır. Daha sonra bu chunklar embeddig işlemine tabi tutulur ve vektör veri tabanında vector olarak saklanır.
Daha iyi anlamak için Hugging Face Pipelinelarının çalışma mantığını inceleyelim
İstemciden gelen prompt tokenize yani encoding işlemine maruz kalır. Text olarak gelen veri vektörlere çevrilir ve LLM modele vektör olarak gönderilir. Model daha önce eğitildiği verileri vektör veri tabanına vektörler olarak kaydetmiştir. Ve yine vektör olarak gelen prompt ile veri setinde vektör olarak tutulan verileri karşılaştırır. [8]
 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/64e08522-0644-4208-92aa-ee9890c80157)
 
Şekil-6 Hugging Face pipelinelarının çalışma şeklini gösteren görsel

Burada KNN ya da ANN algoritmalarını da kullanarak vector veri tabanındaki n-boyutlu kelime vectörleri/embeddinglerine en benzeyen prompt seçilir.
Bulunan sonuç, tokenize(decode) işlemi ile vector formatından tekrar text formatına dönüştürülür.[9]
 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/dd7ca5f1-8d97-4866-8c68-a012eea44b4d)
 
Şekil-7 Vektörler arasında yapılan benzerlik aramasının çalışma şeklini gösteren görsel

 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/a69c7a81-b228-4c37-b217-b8c3e7762b10)
 
Şekil-8 Vektörler veri tabanı kullanan LLM’lerin çalışma şeklini gösteren görsel

Aşağıda Langchain kütüphanesini kullanarak az miktarda veri (25 adet pdf dökümanı) ile veri setinin hazırlanması için kullanılan kod parçası verilmiştir.

```py
import os
from langchain.document_loaders import PyPDFLoader

articles = []

for doc in os.listdir():
  if doc.endswith(".pdf") :
    loader = PyPDFLoader(doc)
    pages = loader.load_and_split()
    articles.append(pages)
```

## Yazılım ürünü tasarımı
Sorulan sorulara yanıt verirken bilginin kaynağına atıf verilmeli.
Model kendisine sorulan sorulara doğru cevaplar vermelidir. Bunun yanı sıra bu cevapları nereden aldığını da link ya da makale adı olarak kullanıcıya göstermelidir. 
Hazır dil modelini uzmanlaştırmak için eklenen pdflerdeki verileri chunklara ayırıp hem bu chunkları vector olarak tutması hem de bu chunkların kaynağını (hangi pdf dosyasının kaçıncı sayfasından alındığını) tutması için Pinecone veri tabanı kullanıldı.
Pinecone, doğal dil işleme (NLP) görevleri için tasarlanmış bir vektör veritabanıdır. Vektör veritabanları, geleneksel veritabanlarından farklı bir şekilde çalışır. Tam eşleşmeler için sorgu yapmak yerine, sorgu ile en çok benzer olan vektörü bulmak için bir benzerlik metriği uygularlar.
Pinecone, NLP görevleri için tasarlandığından, vektörlerini semantik olarak benzer olan verileri gruplamak için kullanır. Bu, kullanıcı sorgularını daha etkili ve anlamlı bir şekilde işlemeyi mümkün kılar.
Pinecone, aşağıdaki adımları izleyerek çalışır:
1.	Veriler, vektörlere dönüştürülür. Bu, metin için kelime vektörleri veya kod için kod vektörleri olabilir.
2.	Vektörler, bir pinecone ağacı adı verilen bir yapıda depolanır. Bu yapı, vektörleri benzerliklerine göre gruplar.
3.	Bir sorgu yapıldığında, sorgu vektörüne benzer olan vektörleri bulmak için pinecone ağacı kullanılır.[10]
 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/188dd869-c546-4d84-8ded-2587a7e4f726)

Şekil-9 Vektörler veri tabanı kullanan LLM’lerin çalışma şeklini gösteren görsel

Aşağıda Pinecone veri tabanını kullanmak için yazılan kodları görebilirsiniz.

```py
!pip install pinecone-client

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="6963f17a-74ba-48d7-b72f-9e8eb4f53b9a",environment="gcp-starter")
chunks = []

for article in articles :
  splitter=CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
  article_chunks = splitter.split_documents(article)
  chunks.append(article_chunks)
```

Sorulan sorulara yanıt verirken Langchain kütüphanesinden VectorDBQA class’ını kullanılması gerekti.
VectorDBQA, Google AI tarafından geliştirilen bir soru-cevap sistemidir. Bu sistem, bir veritabanında saklanan bilgilere erişmek ve bunları kullanarak kullanıcıların sorularını yanıtlamak için bir vektör tabanlı yaklaşım kullanır.
VectorDBQA, bir veritabanında saklanan bilgilerden bir vektör temsili oluşturur. Bu vektör temsili, bilgilerin anlamını ve ilişkilerini yakalar. Kullanıcının sorusu alındığında, bu soru için bir vektör temsili oluşturulur. Bu iki vektör temsili, benzerliklerini belirlemek için karşılaştırılır. İki vektör temsili ne kadar benzerse, kullanıcının sorusunun cevabı veritabanında bulunan bilginin o kadar yakın olduğunu gösterir.
VectorDBQA, aşağıdaki adımları izleyerek çalışır:
1.	Veritabanı ön işleme: Veritabanı, VectorDBQA tarafından kullanılabilir hale gelmek için ön işleme tabi tutulur. Bu işlem, verilerin vektör temsillerine dönüştürülmesini içerir.
2.	Soru ön işleme: Kullanıcının sorusu, VectorDBQA tarafından işlenmek üzere hazırlanır. Bu işlem, sorunun vektör temsiline dönüştürülmesini içerir.
3.	Cevap bulma: Sorunun vektör temsili, veritabanındaki bilginin vektör temsilleriyle karşılaştırılır. En benzer vektör temsili bulunursa, bu bilgi kullanıcının sorusunun cevabı olarak kullanılır.
VectorDBQA, açık uçlu, zorlayıcı ve garip sorular dahil olmak üzere çeşitli türde sorulara yanıt verebilir. Bu sistem, aşağıdakiler de dahil olmak üzere çeşitli uygulamalarda kullanılabilir:
•	Bilgi istemcileri
•	Soru-cevap sistemleri
•	Arama motorları
•	Yapay zeka asistanları [11]
Aşağıda LLM modelin prompt alıp response(cevap) döndürmesi için yazılan kodlar yer almakta

```py
from langchain.chains import VectorDBQA
```

# VectorDBQA Bu class bir çeşit chain class'ıdır.İçerisine vectorstore parametresi de verebildiğimiz için bizim durumumuzda bu kullanılmalıdır.

```py
qa = VectorDBQA.from_chain_type(
        llm=hf,
        chain_type="stuff",
        vectorstore = doc_search,
        return_source_documents=True #Bu parametre modelin verdiği cevabı hangi dökümanları baz alarak verdiğini anlamak adına çok önemli!
    )
prompt = "What is explainable AI? Give me a 15 word answer for a beginner?"

response = qa({"query" : prompt})
print(response)
```

Cevap olarak dönen rerspose aşağıda yer almakta:

```txt
/usr/local/lib/python3.10/dist-packages/langchain/chains/retrieval_qa/base.py:256: UserWarning: `VectorDBQA` is deprecated - please use `from langchain.chains import RetrievalQA`
  warnings.warn(
{'query': 'What is explainable AI? Give me a 15 word answer for a beginner?', 'result': ' Explainable AI (XAI) is a subfield of artificial intelligence (AI) focused on developing techniques and tools to provide insights into the decision-making process of AI models, making them more transparent and trustworthy.', 'source_documents': [Document(page_content='evaluation methods and recommendations for different goals in Explainable AI research.\nAdditional Key Words and Phrases: Explainable artificial intelligence (XAI); human-computer interaction\n(HCI); machine learning; explanation; transparency;\nACM Reference Format:\nSina Mohseni, Niloofar Zarei, and Eric D. Ragan. 2020. A Multidisciplinary Survey and Framework for Design\nand Evaluation of Explainable AI Systems. ACM Trans. Interact. Intell. Syst. 1, 1, Article 1 (January 2020),', metadata={'page': 0.0, 'source': '1811.11839.pdf'}), Document(page_content='evaluation methods and recommendations for different goals in Explainable AI research.\nAdditional Key Words and Phrases: Explainable artificial intelligence (XAI); human-computer interaction\n(HCI); machine learning; explanation; transparency;\nACM Reference Format:\nSina Mohseni, Niloofar Zarei, and Eric D. Ragan. 2020. A Multidisciplinary Survey and Framework for Design\nand Evaluation of Explainable AI Systems. ACM Trans. Interact. Intell. Syst. 1, 1, Article 1 (January 2020),', metadata={'page': 0.0, 'source': '1811.11839.pdf'}), Document(page_content='Doran, D., Schulz, S., & Besold, T. R. (2017). What does explainable AI really mean? A new conceptualization of perspectives. arXiv:1710.00794.HOLZINGER ET AL . 11 of 13', metadata={'page': 10.0, 'source': 'WIREs Data Min   Knowl - 2019 - Holzinger - Causability and explainability of artificial intelligence in medicine.pdf'}), Document(page_content='Doran, D., Schulz, S., & Besold, T. R. (2017). What does explainable AI really mean? A new conceptualization of perspectives. arXiv:1710.00794.HOLZINGER ET AL . 11 of 13', metadata={'page': 10.0, 'source': 'WIREs Data Min   Knowl - 2019 - Holzinger - Causability and explainability of artificial intelligence in medicine.pdf'})]}
```

## Sistemin Nasıl Çalıştığını Gösteren Diyagramlar

 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/4ab6ee41-d6c8-4730-9f7f-828b75f8fe65)
 
Şekil-10 Sistemin Blok Diyagramı

Bir sınıf, durumu (nitelikleri) ve davranışı (işlemleri) içine alan bir kavramı temsil eder. Her özniteliğin bir türü
vardır. Her işlemin bir imzası vardır.

 ![image](https://github.com/elifbeyzatok00/Llama2_LLM_Project/assets/102792446/afc19fef-6c0b-4b6a-bab2-b8d4831e867c)
 
Şekil-11 Kullanım senaryosu (use case) diyagramı

Use Case Diyagramları, iş süreçlerinin yönetilmesi aşamasında ihtiyaç duyulan tüm fonksiyonları,bu fonksiyonları
tetikleyecek aktörleri, fonksiyonlardan etkilenecek aktörleri ve fonksiyonlar arasındaki ilişkileri göstermek amacıyla kullanılmaktadır.

## KAYNAKÇA
A ) İnternet Kaynakları:
1.  Snigdha, APPYPİE, https://www.appypie.com/blog/what-are-large-language-models  , 02 EYLÜL 2023
2. https://chat.openai.com/  Chat-GPT, 02 EYLÜL 2023
3. https://chat.openai.com/  Chat-GPT, 02 EYLÜL 2023
4. https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat HuggingFace, 02 EYLÜL 2023
5.https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat HuggingFace, 02 EYLÜL 2023
6. https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI HuggingFace, 02 EYLÜL 2023
7. https://bard.google.com/chat Bard, 30 ARALIK 2023
8. https://www.youtube.com/watch?v=-T8iDxLMuuk&list=PLTPXxbhUt-YWSR8wtILixhZLF9qB_1yZm&index=12 DATABRİCKS YOUTUBE CHANNEL,  30 Aralık 2023 

9. https://www.youtube.com/watch?v=X5DZL58mBg0&list=PLTPXxbhUt-YWSR8wtILixhZLF9qB_1yZm&index=20 DATABRİCKS YOUTUBE CHANNEL,  30 Aralık 2023
10. https://bard.google.com/chat Bard, 30 ARALIK 2023
11. https://chat.openai.com/  Chat-GPT, 25 ARALIK 2023
