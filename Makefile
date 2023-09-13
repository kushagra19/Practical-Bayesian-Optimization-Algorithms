EXPDIR=./$(shell date +%F)
ZACQ=EI
BUDGET=100
RUNS=8
NOISEMEAN=0.0
NOISEVAR=0.01
NUMRESTARTS=2
RAWSAMPLES=32
RANDOMSEED=42
INIT_EXAMPLES=10
ARGS=$(EXPDIR) $(ZACQ) $(BUDGET) $(RUNS) $(NOISEMEAN) $(NOISEVAR) $(NUMRESTARTS) $(RAWSAMPLES) $(RANDOMSEED) $(INIT_EXAMPLES)

basic:
	chmod +x run.sh

le_branke: basic
# 	./run.sh le_branke max 1 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/le_branke/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

branin: basic
# 	./run.sh branin min 2 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/branin/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

levy: basic
# 	./run.sh levy min 4 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/levy/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

rosenbrock: basic
# 	./run.sh rosenbrock min 4 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/rosenbrock/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

ackley: basic
# 	./run.sh ackley min 5 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/ackley/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

hartmann: basic
# 	./run.sh hartmann min 6 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/hartmann/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

powell: basic
# 	./run.sh powell min 7 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/powell/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

holder_table: basic
# 	./run.sh holder_table min 2 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/holder_table/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

gw: basic
# ./run.sh gw max 40 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/gw/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

Cosine8: basic
# 	./run.sh Cosine8 max 8 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/Cosine8/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

Drop_Wave: basic
# 	./run.sh Drop_Wave min 2 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/Drop_Wave/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

StyblinskiTang: basic
# 	./run.sh StyblinskiTang min 4 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/StyblinskiTang/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

DixonPrice: basic
# 	./run.sh DixonPrice min 5 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/DixonPrice/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

SixHumpCamel: basic
# 	./run.sh SixHumpCamel min 2 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/SixHumpCamel/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

ThreeHumpCamel: basic
# 	./run.sh ThreeHumpCamel min 2 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/ThreeHumpCamel/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

Sphere: basic
# 	./run.sh Sphere min 6 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/Sphere/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

Bukin: basic
# 	./run.sh Bukin min 2 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/Bukin/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

Griewank: basic
# 	./run.sh Griewank min 6 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/Griewank/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

Michalewicz: basic
# 	./run.sh Michalewicz min 10 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/Michalewicz/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

Sum_exp: basic
# 	./run.sh Sum_exp min 6 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/Sum_exp/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret


cart_pole: basic
# 	./run.sh Sum_exp min 6 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/cart_pole/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

all: basic
	python3 post_combine.py -e $(EXPDIR) -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

illustrationNd: basic
	python3 post_process.py -e $(EXPDIR)/illustrationNd/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

pongChangeMountainCar: basic
	python3 post_process.py -e $(EXPDIR)/pongChangeMountainCar/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

lunar_lander: basic
	python3 post_process.py -e $(EXPDIR)/lunar_lander/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

rotation_transformer: basic
# ./run.sh rotation_transformer min 1 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/rotation_transformer/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret









