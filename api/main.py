from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from predictor import predictor
from router import router
from router_laboratorio import router_lab


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title='VargasVet IA',
    description='Microservicio IA veterinario: radiografías (DenseNet-121) + extracción de hemogramas',
    version='2.0.0',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(router)
app.include_router(router_lab)


@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return Response(status_code=204)
