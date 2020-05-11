// Databricks notebook source
// import packages
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.functions.spark_partition_id

// COMMAND ----------

//Define class row
case class row(x1 : Double, x2:Double , y:Double)

//Generate Dataframe
val Data=for (i <- 1 to 1000) yield (row(i.asInstanceOf[Double],(i+5).asInstanceOf[Double],(5*i+2).asInstanceOf[Double]))
val Dataframe=Data.toDF()
Dataframe .show()

// COMMAND ----------

//Defenir N° de la partition
val DF = Dataframe.repartition(10)
var DF_f =DF.withColumn("partition_ID", spark_partition_id)
DF_f.show()

// COMMAND ----------

val x = ArrayBuffer(10.0,55.0,5.0)
val y= ArrayBuffer(5.0,5.0,1.0)

def prod_by_scal(x:ArrayBuffer[Double],n:Double)={
  val s=x.size
  var res=ArrayBuffer[Double]()
  for(i<- 0 until s){
    res+=x(i)*n
  }
  res
}

def prod_scal (x:ArrayBuffer[Double], w:ArrayBuffer[Double]):Double =
{
  (x zip w).map(xy=>xy._1*xy._2).sum  
}

prod_by_scal(x,10)
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

def difference(x:ArrayBuffer[Double],y:ArrayBuffer[Double])={ 
  val s=x.size
  var res=ArrayBuffer[Double]()
  for (i<-0 until s)
     { 
       res+=x(i)-y(i)
     }
   res
}
somme(x,y)
difference(x,y)

// COMMAND ----------

//calcul du gradient d'une seule instance
def grad(x:(ArrayBuffer[Double],Double),w:ArrayBuffer[Double])={
   var res=prod_by_scal(x._1,2*(prod_scal(x._1,w)-x._2)) 
   res  
}

// COMMAND ----------

//calcul du gradient d'une partition(Batch)
def compute_grad_Batch( dataframe:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row],W:ArrayBuffer[Double] )={
  var grad_p = ArrayBuffer.fill(W.length)(0.0)
  var row=dataframe.select($"x1",$"x2",$"y").collect()
  for (j<-row){ 
    var x1=j.get(0).asInstanceOf[Double]
    var x2=j.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =j.get(2).asInstanceOf[Double]
    grad_p= somme(grad_p ,  grad((x,y),W) ) 
  }
  grad_p
}

//Calculer Gradient  en utilisant Batch
var W=ArrayBuffer.fill(2)(0.0)
var gd_global =ArrayBuffer.fill(W.length)(0.0)
var nov_W=W
val alpha = 0.000001
val num_iter = 10
val parts=10
for (k<-1 to num_iter)
{
  for(i<- 0 to parts-1)
  {
    val part_act=DF_f.filter($"partition_ID" === i )
    val grad=compute_grad_Batch(part_act ,W )
    gd_global = somme(gd_global , grad )
  }
  
  gd_global=prod_by_scal(gd_global,1/(DF_f.count.toFloat))
  nov_W =  difference(nov_W ,prod_by_scal(gd_global,alpha )) 
  W=nov_W
  println(W)
}

// COMMAND ----------

//SGD-Parition
def SGD( dataframe:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , alpha:Double , W:ArrayBuffer[Double] )={
  var nov_W  = W
  var row=dataframe.select($"x1",$"x2",$"y").collect()
  for (j<-row){ 
    var x1=j.get(0).asInstanceOf[Double]
    var x2=j.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =j.get(2).asInstanceOf[Double]    
    var gd_inst = grad((x,y),nov_W)
    nov_W =  difference(nov_W ,prod_by_scal(gd_inst,alpha )) 
  }
  nov_W
}

//Calculer le gradient stochastique(SGD)
val parts=10
var W=ArrayBuffer.fill(2)(0.0)
val alpha = 0.000001
val num_iter = 3
for (k<-1 to num_iter){
  for(i<- 0 to parts-1){
    val part_act=DF_f.filter($"partition_ID" === i )
    val nov_W=SGD( part_act , alpha , W )
    W=nov_W
    println(nov_W)
  }
}

// COMMAND ----------

// MiniBatch-Partition
def SGD_miniBatch( dataframe:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , alpha:Double , W:ArrayBuffer[Double] )={
  var nov_W  = W
  var gd_Batch = ArrayBuffer.fill(W.length)(0.0) 
  var rows=dataframe.select($"x1",$"x2",$"y").collect()
  for (j<-rows){ 
    var x1=j.get(0).asInstanceOf[Double]
    var x2=j.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =j.get(2).asInstanceOf[Double]    
    var gd_inst = grad((x,y),nov_W)
    gd_Batch= somme(gd_Batch , gd_inst )
  }
  gd_Batch=prod_by_scal(gd_Batch,1/(dataframe.count.toFloat))
  nov_W =  difference(nov_W ,prod_by_scal(gd_Batch,alpha ))
  nov_W
}

//Calculer SGD en utilisant des MiniBatch
var W=ArrayBuffer.fill(2)(0.0)
val alpha = 0.000001
val num_iter=3
val minibatch_rate=30

for (j<-1 to num_iter){
  for(i<- 0 to parts-1){
    val part_act=DF_f.filter($"partition_ID" === i )
    var samples=part_act.sample(true,minibatch_rate)
    val nov_W=SGD_miniBatch( samples , alpha , W )
    W=nov_W
    println(W)
  }
}


// COMMAND ----------

//MOMENTUM-Partition
def Fast_SGD_Momentum( dataframe:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , alpha:Double, Beta:Double, V:ArrayBuffer[Double], W:ArrayBuffer[Double] )={
  var nov_W  = W
  var nov_V=V
  var gradient = ArrayBuffer.fill(W.length)(0.0) 
  var rows=dataframe.select($"x1",$"x2",$"y").collect()
  for (i<-rows){ 
    var x1=i.get(0).asInstanceOf[Double]
    var x2=i.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =i.get(2).asInstanceOf[Double]    
    var gd_inst = grad((x,y),nov_W)
    nov_V=somme(prod_by_scal(nov_V,Beta),prod_by_scal(gd_inst,alpha))
    nov_W =  difference(nov_W ,nov_V)
  }
  (nov_W,nov_V)
}

//Calculer Fast SGD avec Momentum
var nov_W=ArrayBuffer.fill(2)(0.0)
val alpha = 0.000001
val Beta=0.9

var nov_V =ArrayBuffer.fill(2)(0.0)
val num_iter=3
for (k<-1 to num_iter){
    for(i<- 0 to parts-1 ){
      val part_act=DF_f.filter($"partition_ID" === i )
      val (nvvW,nvvV)=Fast_SGD_Momentum( part_act , alpha ,Beta , nov_V , nov_W)
      nov_W=nvvW
      nov_V=nvvV
      println(nov_W)
   }
}


// COMMAND ----------

//ADAGRAD-Partition
import scala.math.sqrt
def Fast_SGD_Adagrad( dataframe:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , alpha:Double,  W:ArrayBuffer[Double] )=
{
  var nov_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  
  var rows=dataframe.select($"x1",$"x2",$"y").collect()
  for (i<-rows){ 
    var x1=i.get(0).asInstanceOf[Double]
    var x2=i.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =i.get(2).asInstanceOf[Double]    
    var gd_inst = grad((x,y),nov_W)
    var G=prod_scal(gd_inst,gd_inst)+ 1e-8
    var new_alpha = alpha * (1/sqrt(G))
    nov_W=difference(nov_W,prod_by_scal(gd_inst ,new_alpha))
  }
  nov_W
}

//Calculer Fast SGD avec Adagrad 
var nov_W=ArrayBuffer.fill(2)(0.0)
val alpha = 0.0025
var new_alpha=alpha

val num_iter=3
for (j<-1 to num_iter)
{
  for(i<- 0 to parts-1){
      val part_act=DF_f  .filter($"partition_ID" === i )
      val nvvW=Fast_SGD_Adagrad( part_act , alpha , nov_W )
      nov_W=nvvW
      println(nov_W)
  }
}

// COMMAND ----------



