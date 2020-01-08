# optimalstop

## This program is used in oder to solve the optimal stop problem labeled here:
https://en.wikipedia.org/wiki/Optimal_stopping

## This is the original correspondence between myself and the author of the program:
Here is my HACK of a python program to test my method for the optimal stopping problem.  Effectively the method does the following:
1. Assume the skills of the secretary or "fitness" for the job is distributed normally & score each applicant during their interview.
2. Starting at interview #2 calculate each of the following after each interview
  2a. The mean of all applicants interviewed
  2b. The standard deviation of all applicants interviewed
  2c. Bernards estimate of rank (percentile) for the number of applicants left - AND the Bernard's estimate for the number of applicants left minus 1.  Since the Nth Bernard's of N things is an estimate of the percentile of the highest percentile thing, that implies there is a 50% chance of finding the Nth Bernards rank of N things.  Similarly, since the Bernard's rank of the (N-1)th thing of N things is the ESTIMATE of the 2nd highest thing, I calculate what is half way between the Nth Bernards of N things (estimate of highest percentile thing) and the Bernards rank of the penultimate thing (Nth-1) of N things (estimate of the 2nd highest percentile thing).  This average sets my "stopping criteria" halfway between the EXPECTED highest percentile thing (Nth of N) and the 2nd highest EXPECTED thing ([N-1]th of N ).  Some simple algebra of [(N-0.3)/(N+0.4) + ((N-1)-0.3)/(N+0.4)] / 2 gives me (N\**2-0.9*N-0.17)/(N**2-0.2*N-0.24).  That becomes my selection criteria.  Note I use the number of applicants LEFT to interview to calculate my maximum EXPECTED percentile - so as the interview process goes on my percentile selection criterion drops.  That can be observed in the plots attached (each color is a separate random dataset)