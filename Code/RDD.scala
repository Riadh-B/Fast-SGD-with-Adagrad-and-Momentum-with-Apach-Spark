// Databricks notebook source
// import packages
import scala.collection.mutable.ArrayBuffer
import scala.math.sqrt

// COMMAND ----------


def prod_by_scal(x:ArrayBuffer[Double],n:Double)={
  val s=x.size
  var res=ArrayBuffer[Double]()
  for(i<- 0 until s){
    res+=x(i)*n
  }
  res
}
val x = ArrayBuffer(5.0,2.0,5.0)
val y= ArrayBuffer(2.0,1.0,1.0)
prod_by_scal(x,10)


// COMMAND ----------



def prod_scal (x:ArrayBuffer[Double], w:ArrayBuffer[Double]):Double =
{
  (x zip w).map(xy=>xy._1*xy._2).sum
  
}

prod_scal(x,y)

// COMMAND ----------

def somme(x:ArrayBuffer[Double],y:ArrayBuffer[Double])={
  val s=x.size
  var res=new ArrayBuffer[Double]()
  for(i<-0 until s){
    res+=x(i)+y(i)
  }
  res
}
val x = ArrayBuffer(5.0,2.0,5.0)

def difference(x:ArrayBuffer[Double],y:ArrayBuffer[Double])={ 
  val s=x.size
  var res=ArrayBuffer[Double]()
  for (i<-0 until s)
     { 
       res+=x(i)-y(i)
     }
   res
}

somme(x,x)
difference(x,y)

// COMMAND ----------

//calcul du gradient d'une seule instance
def grad(x:(ArrayBuffer[Double],Double),w:ArrayBuffer[Double])={
   var res=prod_by_scal(x._1,2*(prod_scal(x._1,w)-x._2)) 
   res  
}

//calcul du gradient (Batch) d'une partition
def compute_grad_Batch( p:Array[(ArrayBuffer[Double] , Double)],W:ArrayBuffer[Double] )={
  var grad_p = ArrayBuffer.fill(W.length)(0.0)
  for (i <- p){
    grad_p= somme(grad_p , grad(i,W) ) 
  }
  grad_p
}

val x = ArrayBuffer(1.0,2.0,3.0)
var z = ArrayBuffer(3.0,1.0,5.0)

grad((x,2),z) 

// COMMAND ----------

// Génerer des données
val donnes=(1 to 1000).toArray
val instances=donnes.map(x=>(ArrayBuffer(x.asInstanceOf[Double],(x+5).asInstanceOf[Double]),(5*x+2).asInstanceOf[Double]))

// COMMAND ----------

val parts=10
val input_rdd=sc.parallelize(instances,1).repartition(parts)
val partitions=input_rdd.glom.zipWithIndex().map(x=>(x._2,x._1))
var W=ArrayBuffer.fill(2)(0.0)
val alpha = 0.000001
val num_iter = 10
 
var gd_global =ArrayBuffer.fill(W.length)(0.0)
var nov_W=W

//Calculer Gradient  en utilisant Batch
for (k<-1 to num_iter)
{
  for(i<- 0 to parts-1)
  {
    val part_act=partitions.filter(p=>p._1==i)
    val grad=part_act.map(x=>compute_grad_Batch( x._2 ,W )).collect()
    gd_global = somme(gd_global , grad(0) )
  }
  
  gd_global=prod_by_scal(gd_global,1/(instances.length.toFloat))
  nov_W =  difference(nov_W ,prod_by_scal(gd_global,alpha )) 
  W=nov_W
  println(W)
}

// COMMAND ----------

//SGD-Parition
def SGD( p:Array[(ArrayBuffer[Double] , Double)] , alpha:Double , W:ArrayBuffer[Double] )={
  var nov_W  = W
  for (i <- p)
  {
    var grad_inst = grad(i,nov_W)
    nov_W =  difference(nov_W ,prod_by_scal(grad_inst,alpha )) 
  } 
  nov_W
}

//Calculer le gradient stochastique(SGD)
val alpha = 0.000001
val num_iter = 3

for (k<-1 to num_iter)
{
  for(i<- 0 to parts-1)
  {
    val part_act=partitions.filter(p=>p._1==i)
    val nvvW=part_act.map(x=>SGD( x._2 , alpha , W )).collect
    W=nvvW(0)
    println(W)
  }
}

// COMMAND ----------


// MiniBatch-Partition
def SGD_miniBatch( p:Array[(ArrayBuffer[Double] , Double)] , alpha:Double , W:ArrayBuffer[Double] )={
  var nov_W  = W
  var gd_Batch = ArrayBuffer.fill(W.length)(0.0)
  for (i <- p)
  {
    gd_Batch = somme(gd_Batch  , grad(i,nov_W) )  
  }
  nov_W =  difference(nov_W ,prod_by_scal(prod_by_scal(gd_Batch,1/(p.length.toFloat)),alpha ))
  nov_W
}

//Calculer SGD en utilisant des MiniBatch
val minibatch_rate=30
var W=ArrayBuffer.fill(2)(0.0)
val alpha = 0.000001
val num_iter=3
for (k<-1 to num_iter){
  for(i<- 0 to parts-1){

    val part_act=partitions.filter(p=>p._1==i)
    var samples=part_act.sample(true,minibatch_rate)
    val nvvW=samples.map(x=>SGD_miniBatch(x._2 , alpha , W )).collect
    W=nvvW(0)
    println(W)
  }
}


// COMMAND ----------

//MOMENTUM-Partition
def Fast_SGD_Momentum( p:Array[(ArrayBuffer[Double] , Double)] , alpha:Double, Beta:Double, V:ArrayBuffer[Double], W:ArrayBuffer[Double] )={
  var nov_V=V
  var nov_W  = W
  var gd_Mom = ArrayBuffer.fill(2)(0.0)
  for (i <- p){
    var grad_inst= grad(i,nov_W)
    nov_V=somme(prod_by_scal(nov_V,Beta),prod_by_scal(grad_inst,alpha))
    nov_W =  difference(nov_W ,nov_V)
  }
  (nov_W,nov_V)
}

//Calculer Fast SGD avec Momentum
val alpha = 0.000001
val Beta=0.9
var W=ArrayBuffer.fill(2)(0.0)
var nov_V =ArrayBuffer.fill(2)(0.0)
val num_iter=3

for (k<-1 to num_iter){  
    for(i<- 0 to parts-1){
      val part_act=partitions.filter(p=>p._1==i)
      val nvvW=part_act.map(x=>Fast_SGD_Momentum( x._2 , alpha ,Beta , nov_V , W )).collect
      W=nvvW(0)._1
      nov_V=nvvW(0)._2
      println(W)
   }
}

// COMMAND ----------

  //ADAGRAD-Partition
def Fast_SGD_Adagrad( p:Array[(ArrayBuffer[Double] , Double)] , alpha:Double,  W:ArrayBuffer[Double] )={
  var nov_W  = W
  for (i <- p){ 
    var grad_inst= grad(i,nov_W)
    var G=prod_scal(grad_inst,grad_inst)+ 1e-8
    var nov_alpha = alpha * (1/sqrt(G))
    nov_W=difference(nov_W,prod_by_scal(grad_inst,nov_alpha))
  }
  nov_W
}

//Calculer Fast SGD avec Adagrad 
var W=ArrayBuffer.fill(2)(0.0)
val alpha = 0.001
var nov_alpha=alpha

val num_iter=3
for (k<-1 to num_iter){
  for(i<- 0 to parts-1){
      val part_act=partitions.filter(p=>p._1==i)
      val nvvW=part_act.map(x=>Fast_SGD_Adagrad( x._2 , alpha , W )).collect
      W=nvvW(0)
      println(W)
  }

}

