import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {NeuralNetwork,random} from '../dist/model.js';
import {sample,split,parseCSV} from '../dist/data.js';
test('preserves original forward and backward updates at learning rate 0.01',()=>{
 const rng=random(42),math=Object.create(Math);math.random=rng;
 const ctx=vm.createContext({Math:math,console:{log(){}}});
 vm.runInContext(readFileSync(new URL('../original/utils.js',import.meta.url),'utf8')+'\n'+readFileSync(new URL('../original/neural.js',import.meta.url),'utf8')+'\nthis.original=new Neural_Network([2,4,3,1]);',ctx);
 const model=new NeuralNetwork(),points=sample('clusters',20,7);ctx.inputs=points.map(p=>[p.x,p.y]);ctx.targets=points.map(p=>[p.label]);
 assert.equal(model.predict([.3,.8]),ctx.original.predict([.3,.8])[0]);
 vm.runInContext('original.fit(inputs,targets,3)',ctx);for(let i=0;i<3;i++)model.trainEpoch(points);
 for(const p of points)assert.ok(Math.abs(model.predict([p.x,p.y])-ctx.original.predict([p.x,p.y])[0])<1e-14);
});
test('learns held-out clusters, with repeatable seeded training',()=>{const {train,test:held}=split(sample('clusters'));const a=new NeuralNetwork(),b=new NeuralNetwork();const initial=a.metrics(train).loss;for(let i=0;i<3000;i++){a.trainEpoch(train);b.trainEpoch(train);}assert.ok(a.metrics(train).loss<initial*.2);assert.ok(a.metrics(held).accuracy>=.95);assert.equal(a.predict([.3,.3]),b.predict([.3,.3]));});
test('can learn nonlinear XOR with configurable rate',()=>{const points=[{x:0,y:0,label:0},{x:0,y:1,label:1},{x:1,y:0,label:1},{x:1,y:1,label:0}];const m=new NeuralNetwork([2,4,3,1],.3,42);for(let i=0;i<15000;i++)m.trainEpoch(points);assert.equal(m.metrics(points).accuracy,1);});
test('test set is stratified, disjoint, and not needed for tiny datasets',()=>{const points=sample('circle');const {train,test:held}=split(points);assert.equal(train.length,80);assert.equal(held.length,20);assert.equal(held.filter(p=>p.label===1).length,10);assert.ok(!train.some(p=>held.some(q=>p.x===q.x&&p.y===q.y)));assert.equal(split(points.slice(0,4)).test.length,0);});
test('CSV rejects missing, nonfinite, out-of-range and invalid labels',()=>{assert.equal(parseCSV('x,y,label\n0,1,0\n1,0,1').length,2);for(const csv of ['x,y,label\n,1,0\n1,0,1','0,1,2\n1,0,1','NaN,1,0\n1,0,1','1.1,0,1\n0,1,0',''])assert.throws(()=>parseCSV(csv));assert.throws(()=>new NeuralNetwork([2,99,1]));assert.throws(()=>new NeuralNetwork().predict([NaN,0]));});
