import fitz
import uuid
import qrcode
import hashlib
import json, os
import traceback
from _base import gen
from encoder import sbert
from flask_cors import CORS
from _db_qdrant import searchk
from _pdf_processing import to_image
from flask_autoindex import AutoIndex
from flask import Flask, request, jsonify
from sentence_transformers import CrossEncoder
from _db_qdrant import create_collection, drop
from utils import list_pdf, rerank, push_slack, get_id_from_tasks
from indexer import vectorisation, insert, get_st
from _record import (authenticate, is_user_exists, create_user, 
                        create_session, check_session,
                        _create_product, _remove_product, _select_products,
                        _check_index, _select_documents, _create_document,
                        create_questions, _select_questions,
                        _select_tasks, _reprioritize,
                        _create_ip, _select_ip
                        )

url_backend = "http://localhost:7777"
url_search = "http://localhost:3000"

url_backend = "https://e393-54-197-22-94.ngrok-free.app"
url_search = "https://heuristic-cs.vercel.app"

if os.path.isdir("/home/ubuntu/cx/backend/assets"):
    ppath = "/home/ubuntu/cx/backend/assets" # step this path to your OS and accordingly from where to you start it
else:
    ppath = "/Users/bm7/space/df/cx/back/assets"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})
AutoIndex(app, browse_root="./assets")

@app.route("/index", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        data = request.data.decode()
        loaded = json.loads(data)
        _indexed = _check_index(loaded["_product_id"])
        return { "session": "indexed" if _indexed > 0 else "non indexed", "size": _indexed }
    else:
        return { "session": "must POST request" }
    
def indexing(pdf, pdt):
    print("indexing...")
    
    docs = [ pdf ]
    # drop("cx")

    create_collection("cx", dim=768)

    print("->Preprocess and Vectorizing... {} documents.".format(len(docs)))
    vs = vectorisation(docs=docs, pdt=pdt)
    print("DONE")
    insert(vs, size=768, col="cx")
    print("->Indexing done")
    return vs

@app.route("/upload", methods=["POST", "GET"])
def upload():
    def hash(pdt):
        return (hashlib.md5((pdt+"5gz").encode())).hexdigest()
    
    if request.method == "POST":
        files = request.files
        PDT_filename = files["file"].filename
        PDT_id = request.form["product"]
        company = request.form["company"]

        # -> Store PDF product document
        files["file"].save("./assets/"+PDT_filename)

        # -> Generate QR-code form the product-ID and save the image
        uid =  str(hash(PDT_id))
        qr_image_path = uid+".png"
        PDT_qr_path = "./assets/"+qr_image_path
        qr = qrcode.make( url_search+"/"+company+"/"+PDT_id+"/search" )
        qr.save(PDT_qr_path)

        # -> Vectorize the document PDF -> {filename,context,page}
        vs = indexing("./assets/"+PDT_filename, PDT_id)

        # -> Store filename in table
        _create_document(PDT_filename, PDT_id)

        # -> Return image-Qr qr-path
        return { "session": "created" , "qr_code": qr_image_path}
    else:
        return { "session": "must POST request" }
    
@app.route('/gener', methods = ['POST', 'GET'])
def search():
    if request.method == 'POST':
        data = request.data.decode()
        data = json.loads(data)
        query, product, company = data["query"], data["product"], data["company"]


        query_vector = sbert([query])[0]
        result = searchk(query_vector, topk=5, pdt=product)

        contexts = [hit.payload["passage"] for hit in result]
        ranked = list(reversed(sorted(rerank(model, query, contexts))))
        print("after")

        answer = gen(ranked[0], query)
        if answer != "Not found":
            # Store data in TiDB for HTAP operations
            create_questions(query, company, product, solved="yes")

            # Operations on the replicated tables
            # 1. Select all tasks that have a level greater than 0
            tasks = _select_tasks(nb="1", prio=1)
            # 2. Compute the similarity score between the consumer query and all retrieved task
            ranked_tasks = list(reversed(sorted(rerank(model, query, [ task[1] for task in tasks ]))))
            task_to_reprioritze = ranked_tasks[0][1]
            task_id, task_level = get_id_from_tasks(task_to_reprioritze, tasks)
            # Let's reprioritize the task
            _reprioritize(task_id, str(int(task_level)+1))
        else:
            # Store data in TiDB for HTAP operations
            create_questions(query, company, product, solved="no")


        hits = []
        for id, hit in enumerate(result[:]):

            context = ranked[0]
            _random = get_st()+".png"

            _path = os.path.join("./assets", _random)
            if (hit.payload["passage"]==context[1]):
                
                hits.append({
                        "path": hit.payload["path"],
                        "answer": answer,
                        "score": str(hit.score),
                        "id": hit.id,
                        "img": "/"+_random,
                        "pdf": url_backend+"/"+hit.payload["path"].split("/")[-1],
                        "page": hit.payload["page"]
                    })
                to_image(doc=fitz.open(hit.payload["path"]), page=hit.payload["page"],image_name= _path, answer="")
                push_slack(f"Query:{query}\nAnswer:{answer}\n---")
                print("Problem happen after, saving image...", all([ type(v) != float for k, v in hits[0].items() ]))
        return hits
    else:
        return { "session": "must POST request" }

if __name__ == '__main__':
    create_collection("cx")
    model = CrossEncoder("cross-encoder/msmarco-MiniLM-L12-en-de-v1", max_length=512)
    app.run(debug = True, host="0.0.0.0", port="7777", use_reloader=False, threaded=True)
