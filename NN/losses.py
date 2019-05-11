import torch.autograd as autograd
import torch
import utils as utils
import torch.nn as nn


class LossReserveFixedLazySecondPrice():
    def __init__(self,reserve,distrib,nb_opponents=1):
        self.name = "LossReserveFixedLazySecondPrice"
        self.reserve = reserve
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(100*(output - self.reserve))
          winning = self.distrib.cdf(output)**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
                        virtual_value_eval,winning),indicator))
          return loss

class LossReserveFixedEagerSecondPrice():
    def __init__(self,reserve,distrib,nb_opponents=1):
        self.name = "LossReserveFixedEagerSecondPrice"
        self.reserve = reserve
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(100*(output - self.reserve))
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
                self.distrib.cdf(output))**self.nb_opponents

          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
                        virtual_value_eval,winning),indicator))
          return loss

class LossMonopolyReserveLazySecondPrice():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "LossMonopolyReserveLazySecondPrice"
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                        create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(1000*(virtual_value_eval-0.001))
          winning = self.distrib.cdf(output)**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
           virtual_value_eval,torch.min(self.distrib.cdf(output)**self.nb_opponents,torch.tensor(1.0))),indicator))
          return loss

class LossMonopolyReserveEagerSecondPrice():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "LossMonopolyReserveEagerSecondPrice"
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                        create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(1000*(virtual_value_eval-0.001))
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(output))**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
           virtual_value_eval,torch.min(winning,torch.tensor(1.0))),indicator))
          return loss

class lossMyersonAuction():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "LossPersonalizedReserve"
        self.nb_opponents = nb_opponents
        self.distrib =distrib

    def eval(self,net,input,size_batch,nb_opponents=3):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)

          indicator = torch.sigmoid(1000*(virtual_value_eval-0.000001))
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(self.distrib.inverse_virtual_value(virtual_value_eval)))**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
                virtual_value_eval,torch.min(winning,\
                    torch.tensor(1.0))),indicator))
          return loss


class lossBoostedSecondPriceLinearFit():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "BoostedSecondPriceLinear"
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)

          output_grad = autograd.grad(torch.sum(output),input,\
                    retain_graph=True,create_graph=True)[0]

          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(1000*virtual_value_eval)*torch.sigmoid(1000*output)
          # fit a corresponding to the linear fit above the reserve price
          mean_psi = torch.sum(virtual_value_eval*indicator)/torch.sum(indicator)
          mean_value = torch.sum(input*indicator)/torch.sum(indicator)
          min_fit = torch.min(torch.where((indicator>0.5),input,torch.tensor(0.95)))
          a = mean_psi / (mean_value-min_fit)
          virtual_value_fit = a*input
          #compute probability of winning
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),self.distrib.cdf(virtual_value_fit/self.distrib.boost))**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - torch.max(virtual_value_fit,torch.tensor(0.0)),\
          torch.min(winning,\
                    torch.tensor(1.0))),indicator))
          return loss

class lossBoostedSecondPriceAffineFit():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "BoostedSecondPriceAffine"
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)

          output_grad = autograd.grad(torch.sum(output),input,\
                    retain_graph=True,create_graph=True)[0]

          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)

          #fit a,b corresponding to boosted second price
          mean_psi = torch.mean(virtual_value_eval)
          mean_bid = torch.mean(output)
          a = torch.sum(torch.mul(virtual_value_eval - mean_psi,output_grad - mean_bid)) \
            /torch.sum(torch.mul(output - mean_bid,output - mean_bid))
          b = mean_psi - a * mean_bid
          virtual_value_fit = a*output + b

          #use affine fit to define the reserve price
          indicator = torch.sigmoid(1000*(virtual_value_fit))

          #compute the probability of winning
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(self.distrib.inverse_virtual_value(virtual_value_fit)))**self.nb_opponents

          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - virtual_value_fit,\
          torch.min(winning,torch.tensor(1.0))),indicator))
          return loss
